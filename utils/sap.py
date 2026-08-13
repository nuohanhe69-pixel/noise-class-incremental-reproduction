"""Small, side-effect-free mathematical primitives for SAP.

The scaling follows the official SAP implementation at commit
6e776ef71324d7c44afea2130b6738debdcbae8b. Model hooks, reference
selection, and task-boundary orchestration intentionally live elsewhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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


def build_sap_projection_from_gram(gram: Tensor, scale: float) -> Tensor:
    """Build ``Mr`` from ``X.T @ X`` using the official SAP scaling rule."""
    symmetric_gram = _validate_gram(gram, scale)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_gram)

    # Round-off can produce tiny negative eigenvalues for a PSD Gram matrix.
    energy = eigenvalues.clamp_min(0)
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
