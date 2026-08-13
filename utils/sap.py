"""Small, side-effect-free mathematical primitives for SAP.

The scaling follows the official SAP implementation at commit
6e776ef71324d7c44afea2130b6738debdcbae8b. Model hooks, reference
selection, and task-boundary orchestration intentionally live elsewhere.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


RESNET18_LATE_STAGE_CONVS = (
    'layer3.0.conv1',
    'layer3.0.conv2',
    'layer3.0.shortcut.0',
    'layer3.1.conv1',
    'layer3.1.conv2',
    'layer4.0.conv1',
    'layer4.0.conv2',
    'layer4.0.shortcut.0',
    'layer4.1.conv1',
    'layer4.1.conv2',
)


@dataclass(frozen=True)
class SAPGramStats:
    gram: Tensor
    available_patches: int
    sampled_patches: int
    patch_dimension: int


@dataclass(frozen=True)
class SAPLayerProjectionStats:
    available_patches: int
    sampled_patches: int
    patch_dimension: int
    relative_weight_delta: float


def _unwrap_parallel_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def resolve_resnet18_sap_layers(model: nn.Module) -> OrderedDict[str, nn.Conv2d]:
    """Resolve the ten approved ResNet18 layer3/layer4 convolution modules."""
    backbone = _unwrap_parallel_model(model)
    resolved = OrderedDict()
    for name in RESNET18_LATE_STAGE_CONVS:
        try:
            layer = backbone.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f'model does not provide the required SAP layer {name!r}') from error
        if not isinstance(layer, nn.Conv2d):
            raise TypeError(f'required SAP layer {name!r} is not a Conv2d')
        if layer.groups != 1:
            raise ValueError(f'required SAP layer {name!r} is a grouped convolution')
        resolved[name] = layer
    return resolved


def conv2d_input_to_patches(inputs: Tensor, conv: nn.Conv2d) -> Tensor:
    """Return one flattened input patch per output spatial position."""
    if inputs.ndim != 4:
        raise ValueError(f'expected NCHW Conv2d inputs, got shape {tuple(inputs.shape)}')
    if inputs.shape[1] != conv.in_channels:
        raise ValueError(
            f'input has {inputs.shape[1]} channels but convolution expects {conv.in_channels}'
        )
    if conv.groups != 1:
        raise ValueError('grouped convolutions are not supported by SAP patch extraction')

    # F.unfold returns [batch, channels * kernel_height * kernel_width, locations].
    unfolded = F.unfold(
        inputs,
        kernel_size=conv.kernel_size,
        dilation=conv.dilation,
        padding=conv.padding,
        stride=conv.stride,
    )
    return unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])


def _validate_gram(gram: Tensor, scale: float) -> Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f'gram matrix must be square, got shape {tuple(gram.shape)}')
    if not gram.is_floating_point():
        raise TypeError('gram matrix must use a floating-point dtype')
    if not torch.isfinite(gram).all():
        raise ValueError('gram matrix contains NaN or Inf')
    if scale <= 0:
        raise ValueError('SAP scale must be positive')

    symmetric_gram = (gram + gram.transpose(0, 1)) * 0.5
    if torch.trace(symmetric_gram) <= 0:
        raise ValueError('SAP requires positive activation energy')
    return symmetric_gram


def _sap_importance(energy: Tensor, scale: float) -> Tensor:
    ratios = energy / energy.sum()
    return scale * ratios / ((scale - 1.0) * ratios + 1.0)


def build_sap_projection_from_gram(
    gram: Tensor,
    scale: float,
    *,
    max_rank: int | None = None,
) -> Tensor:
    """Build ``Mr`` from ``X.T @ X`` using the official SAP scaling rule."""
    symmetric_gram = _validate_gram(gram, scale)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_gram)

    # Round-off can produce tiny negative eigenvalues for a PSD Gram matrix.
    energy = eigenvalues.clamp_min(0)
    if max_rank is not None:
        if max_rank <= 0:
            raise ValueError('max_rank must be positive')
        keep = min(max_rank, energy.numel())
        energy = energy[-keep:]
        eigenvectors = eigenvectors[:, -keep:]
    importance = _sap_importance(energy, scale)
    projection = (eigenvectors * importance.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    return (projection + projection.transpose(0, 1)) * 0.5


def build_sap_projection_from_patches(patches: Tensor, scale: float) -> Tensor:
    """Direct-SVD reference implementation used to verify the Gram path."""
    if patches.ndim != 2:
        raise ValueError(f'patch matrix must be 2-D, got shape {tuple(patches.shape)}')
    if not patches.is_floating_point():
        raise TypeError('patch matrix must use a floating-point dtype')
    if not torch.isfinite(patches).all():
        raise ValueError('patch matrix contains NaN or Inf')
    if scale <= 0:
        raise ValueError('SAP scale must be positive')

    _, singular_values, vh = torch.linalg.svd(patches, full_matrices=False)
    energy = singular_values.square()
    if energy.sum() <= 0:
        raise ValueError('SAP requires positive activation energy')
    importance = _sap_importance(energy, scale)
    directions = vh.transpose(0, 1)
    projection = (directions * importance.unsqueeze(0)) @ directions.transpose(0, 1)
    return (projection + projection.transpose(0, 1)) * 0.5


def project_conv2d_weight(weight: Tensor, projection: Tensor) -> Tensor:
    """Apply official SAP pre-projection: ``W_flat @ Mr.T``."""
    if weight.ndim != 4:
        raise ValueError(f'Conv2d weight must be 4-D, got shape {tuple(weight.shape)}')
    flat_weight = weight.flatten(1)
    expected_shape = (flat_weight.shape[1], flat_weight.shape[1])
    if projection.shape != expected_shape:
        raise ValueError(
            f'projection shape {tuple(projection.shape)} does not match '
            f'flattened Conv2d input dimension {expected_shape[0]}'
        )
    if projection.device != weight.device:
        raise ValueError('projection and weight must be on the same device')

    projected = flat_weight @ projection.to(dtype=weight.dtype).transpose(0, 1)
    return projected.reshape_as(weight)


def collect_conv2d_input_gram(
    model: nn.Module,
    batches: Iterable[Tensor],
    layer_name: str,
    *,
    total_images: int,
    max_patches: int,
    seed: int,
) -> SAPGramStats:
    """Collect a balanced, deterministic input-patch Gram matrix with one pre-hook."""
    if total_images <= 0:
        raise ValueError('total_images must be positive')
    if max_patches <= 0:
        raise ValueError('max_patches must be positive')

    backbone = _unwrap_parallel_model(model)
    try:
        layer = backbone.get_submodule(layer_name)
    except AttributeError as error:
        raise ValueError(f'model does not provide layer {layer_name!r}') from error
    if not isinstance(layer, nn.Conv2d):
        raise TypeError(f'layer {layer_name!r} is not a Conv2d')

    quota, extra = divmod(max_patches, total_images)
    gram = None
    available_patches = 0
    sampled_patches = 0
    image_offset = 0

    def capture_inputs(module: nn.Module, args: tuple[Tensor, ...]) -> None:
        nonlocal gram, available_patches, sampled_patches, image_offset
        if len(args) == 0:
            raise RuntimeError(f'layer {layer_name!r} received no positional input')
        inputs = args[0].detach()
        unfolded = F.unfold(
            inputs,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        ).transpose(1, 2)
        patches_per_image = unfolded.shape[1]
        available_patches += inputs.shape[0] * patches_per_image

        selected_rows = []
        for local_index, image_patches in enumerate(unfolded):
            image_index = image_offset + local_index
            if image_index >= total_images:
                raise ValueError('batches contain more images than total_images')
            image_quota = quota + int(image_index < extra)
            take = min(image_quota, patches_per_image)
            if take == 0:
                continue
            generator = torch.Generator(device='cpu').manual_seed(int(seed) + image_index)
            indices = torch.randperm(patches_per_image, generator=generator)[:take]
            selected_rows.append(image_patches.index_select(0, indices.to(image_patches.device)))

        image_offset += inputs.shape[0]
        if selected_rows:
            selected = torch.cat(selected_rows, dim=0)
            batch_gram = selected.transpose(0, 1) @ selected
            gram = batch_gram if gram is None else gram + batch_gram
            sampled_patches += selected.shape[0]

    training_states = {module: module.training for module in model.modules()}
    handle = layer.register_forward_pre_hook(capture_inputs)
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                if not isinstance(batch, Tensor):
                    raise TypeError('each SAP activation batch must be a prepared input Tensor')
                model(batch)
    finally:
        handle.remove()
        for module, was_training in training_states.items():
            module.training = was_training

    if image_offset != total_images:
        raise ValueError(f'batches contain {image_offset} images but total_images is {total_images}')
    if gram is None or sampled_patches == 0:
        raise ValueError('no activation patches were sampled')
    return SAPGramStats(
        gram=gram,
        available_patches=available_patches,
        sampled_patches=sampled_patches,
        patch_dimension=gram.shape[0],
    )


def apply_resnet18_sap_projections(
    model: nn.Module,
    projections: Mapping[str, Tensor],
) -> OrderedDict[str, float]:
    """Project only the approved late-stage convolution weights in place."""
    invalid = set(projections) - set(RESNET18_LATE_STAGE_CONVS)
    if invalid:
        raise ValueError(f'{sorted(invalid)} is not an allowed SAP target')
    missing = set(RESNET18_LATE_STAGE_CONVS) - set(projections)
    if missing:
        raise ValueError(f'missing SAP projections for {sorted(missing)}')

    layers = resolve_resnet18_sap_layers(model)
    prepared_projections = OrderedDict()
    for name, layer in layers.items():
        weight = layer.weight.detach()
        projection = projections[name]
        expected_dimension = weight.flatten(1).shape[1]
        if projection.shape != (expected_dimension, expected_dimension):
            raise ValueError(
                f'projection shape {tuple(projection.shape)} does not match '
                f'flattened Conv2d input dimension {expected_dimension} for layer {name!r}'
            )
        if not projection.is_floating_point():
            raise TypeError(f'SAP projection for layer {name!r} must use a floating-point dtype')
        if not torch.isfinite(projection).all():
            raise ValueError(f'SAP projection contains NaN or Inf for layer {name!r}')
        prepared_projections[name] = projection.to(device=weight.device, dtype=weight.dtype)

    deltas = OrderedDict()
    with torch.no_grad():
        for name, layer in layers.items():
            before = layer.weight.detach().clone()
            projected = project_conv2d_weight(before, prepared_projections[name])
            if not torch.isfinite(projected).all():
                raise ValueError(f'SAP projection produced NaN or Inf for layer {name!r}')
            layer.weight.copy_(projected)
            denominator = before.norm().clamp_min(torch.finfo(before.dtype).eps)
            deltas[name] = ((projected - before).norm() / denominator).item()
    return deltas


def project_resnet18_from_reference_batches(
    source_model: nn.Module,
    candidate_model: nn.Module,
    batch_factory: Callable[[], Iterable[Tensor]],
    *,
    total_images: int,
    max_patches: int,
    scale: float,
    seed: int,
) -> OrderedDict[str, SAPLayerProjectionStats]:
    """Project a candidate layer-by-layer from one unchanged source model."""
    source_layers = resolve_resnet18_sap_layers(source_model)
    candidate_layers = resolve_resnet18_sap_layers(candidate_model)
    stats = OrderedDict()

    for layer_index, name in enumerate(RESNET18_LATE_STAGE_CONVS):
        gram_stats = collect_conv2d_input_gram(
            source_model,
            batch_factory(),
            name,
            total_images=total_images,
            max_patches=max_patches,
            seed=int(seed) + layer_index * total_images,
        )
        projection = build_sap_projection_from_gram(
            gram_stats.gram,
            scale,
            max_rank=gram_stats.sampled_patches,
        )

        source_weight = source_layers[name].weight.detach()
        candidate_layer = candidate_layers[name]
        if candidate_layer.weight.shape != source_weight.shape:
            raise ValueError(f'source and candidate weights differ for layer {name!r}')
        projected = project_conv2d_weight(source_weight, projection)
        if not torch.isfinite(projected).all():
            raise ValueError(f'SAP projection produced NaN or Inf for layer {name!r}')
        denominator = source_weight.norm().clamp_min(torch.finfo(source_weight.dtype).eps)
        relative_delta = ((projected - source_weight).norm() / denominator).item()
        with torch.no_grad():
            candidate_layer.weight.copy_(projected)

        stats[name] = SAPLayerProjectionStats(
            available_patches=gram_stats.available_patches,
            sampled_patches=gram_stats.sampled_patches,
            patch_dimension=gram_stats.patch_dimension,
            relative_weight_delta=relative_delta,
        )
        del gram_stats, projection, projected

    return stats
