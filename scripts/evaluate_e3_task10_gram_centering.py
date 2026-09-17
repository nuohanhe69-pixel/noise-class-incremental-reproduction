#!/usr/bin/env python3
"""Offline E3 evaluation of Task10 Gram centering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.sap import (  # noqa: E402
    build_sap_projection_from_gram,
    project_linear_weight,
    resolve_classifier_module,
)
from main import initialize  # noqa: E402
from models.aer_sap import AerSap  # noqa: E402
from utils.checkpoints import mammoth_load_checkpoint  # noqa: E402
import scripts.evaluate_e1_current_block_hybrid as e1  # noqa: E402


SOURCE_RUN_ID = 'b68ebc16-071c-41f9-8011-79bb8d4e2334'
EXPECTED_TASK_COUNT = 10
EXPECTED_LAST_TASK_ID = 9
EXPECTED_SCALE = 3000.0


def validate_task_layout(dataset) -> int:
    task_count = int(dataset.N_TASKS)
    last_task_id = task_count - 1
    if task_count != EXPECTED_TASK_COUNT or last_task_id != EXPECTED_LAST_TASK_ID:
        raise ValueError(
            'E3 requires N_TASKS == 10 and zero-based last_task_id == 9'
        )
    return last_task_id


def _summary(values: Tensor) -> dict[str, float]:
    return {
        'min': float(values.min().item()),
        'median': float(values.median().item()),
        'mean': float(values.mean().item()),
        'max': float(values.max().item()),
    }


def _delta_stats(actual: Tensor, expected: Tensor) -> dict[str, float | bool]:
    if actual.shape != expected.shape:
        raise AssertionError(
            f'tensor shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}'
        )
    difference = actual - expected
    denominator = expected.norm().clamp_min(torch.finfo(expected.dtype).eps)
    return {
        'exact': bool(torch.equal(actual, expected)),
        'max_abs_delta': float(difference.abs().max().item()),
        'relative_delta': float((difference.norm() / denominator).item()),
    }


def _gram_spectrum_diagnostics(gram: Tensor) -> dict[str, float]:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise ValueError('Gram diagnostics require a non-empty square matrix')
    if not gram.is_floating_point() or not torch.isfinite(gram).all():
        raise ValueError('Gram diagnostics require a finite floating-point matrix')
    symmetric_gram = (gram + gram.transpose(0, 1)) * 0.5
    energy = torch.linalg.eigvalsh(symmetric_gram).clamp_min(0)
    total_energy = energy.sum()
    if total_energy <= 0:
        raise ValueError('Gram diagnostics require positive spectral energy')
    energy_ratios = energy / total_energy
    positive_ratios = energy_ratios[energy_ratios > 0]
    effective_rank = torch.exp(
        -(positive_ratios * positive_ratios.log()).sum(),
    )
    top_count = min(10, energy_ratios.numel())
    diagnostics = {
        'trace': float(gram.trace().item()),
        'top1_energy_ratio': float(energy_ratios[-1].item()),
        'top10_energy_ratio': float(energy_ratios[-top_count:].sum().item()),
        'effective_rank': float(effective_rank.item()),
    }
    if not torch.isfinite(torch.tensor(list(diagnostics.values()))).all():
        raise ValueError('Gram diagnostics contain NaN or Inf')
    return diagnostics


def _projection_spectrum_diagnostics(projection: Tensor) -> dict[str, float]:
    if (
        projection.ndim != 2
        or projection.shape[0] != projection.shape[1]
        or projection.shape[0] == 0
    ):
        raise ValueError('projection diagnostics require a non-empty square matrix')
    if not projection.is_floating_point() or not torch.isfinite(projection).all():
        raise ValueError(
            'projection diagnostics require a finite floating-point matrix'
        )
    symmetric_projection = (
        projection + projection.transpose(0, 1)
    ) * 0.5
    eigenvalues = torch.linalg.eigvalsh(symmetric_projection)
    identity = torch.eye(
        projection.shape[0],
        device=projection.device,
        dtype=projection.dtype,
    )
    diagnostics = {
        'trace': float(projection.trace().item()),
        'mean_eigenvalue': float(eigenvalues.mean().item()),
        'median_eigenvalue': float(eigenvalues.median().item()),
        'frobenius_distance_to_identity_ratio': float(
            ((projection - identity).norm() / identity.norm()).item()
        ),
    }
    if not torch.isfinite(torch.tensor(list(diagnostics.values()))).all():
        raise ValueError('projection diagnostics contain NaN or Inf')
    return diagnostics


def build_task10_centered_geometry(
    x_global: Tensor,
    trusted_task_ids: Tensor,
    *,
    last_task_id: int,
    saved_uncentered_gram: Tensor,
    norm_tolerance: float = 1e-5,
) -> dict:
    if x_global.ndim != 2 or not x_global.is_floating_point():
        raise ValueError('X_global must be a floating-point matrix')
    if not torch.isfinite(x_global).all():
        raise ValueError('X_global contains NaN or Inf')
    if trusted_task_ids.ndim != 1 or trusted_task_ids.shape[0] != x_global.shape[0]:
        raise ValueError('trusted task ids must align one-to-one with X_global rows')
    if trusted_task_ids.is_floating_point():
        raise TypeError('trusted task ids must use an integer dtype')

    feature_dim = x_global.shape[1]
    expected_gram_shape = (feature_dim, feature_dim)
    if saved_uncentered_gram.shape != expected_gram_shape:
        raise ValueError('saved uncentered Gram shape does not match X_global')
    if not saved_uncentered_gram.is_floating_point():
        raise TypeError('saved uncentered Gram must use a floating-point dtype')
    saved_uncentered_gram = saved_uncentered_gram.to(
        device=x_global.device,
        dtype=x_global.dtype,
    )
    if not torch.isfinite(saved_uncentered_gram).all():
        raise ValueError('saved uncentered Gram contains NaN or Inf')

    task_mask = trusted_task_ids.to(x_global.device) == int(last_task_id)
    x_task = x_global[task_mask]
    if x_task.shape[0] == 0:
        raise ValueError('Task10 trusted feature set is empty')

    row_norms = x_task.norm(p=2, dim=1)
    torch.testing.assert_close(
        row_norms,
        torch.ones_like(row_norms),
        rtol=norm_tolerance,
        atol=norm_tolerance,
    )

    task_mean = x_task.mean(dim=0, keepdim=True)
    centered_features = x_task - task_mean
    centered_mean_abs_max = float(
        centered_features.mean(dim=0).abs().max().item()
    )
    if centered_mean_abs_max > norm_tolerance:
        raise AssertionError('centered Task10 feature mean is not near zero')
    centered_gram = centered_features.transpose(0, 1) @ centered_features
    if not torch.isfinite(centered_gram).all() or centered_gram.trace() <= 0:
        raise ValueError('centered Gram must have finite positive energy')

    sample_count = int(x_task.shape[0])
    expected_removed_mean = sample_count * task_mean.transpose(0, 1) @ task_mean
    actual_removed_mean = saved_uncentered_gram - centered_gram
    torch.testing.assert_close(
        actual_removed_mean,
        expected_removed_mean,
        rtol=1e-4,
        atol=1e-5,
    )
    expected_centered_trace = (
        saved_uncentered_gram.trace()
        - sample_count * task_mean.square().sum()
    )
    torch.testing.assert_close(
        centered_gram.trace(),
        expected_centered_trace,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        centered_gram,
        centered_gram.transpose(0, 1),
        rtol=0,
        atol=1e-6,
    )

    diagnostics = {
        'task_id': int(last_task_id),
        'sample_count': sample_count,
        'feature_dim': int(feature_dim),
        'row_norm_before_centering': _summary(row_norms),
        'centered_row_norm': _summary(centered_features.norm(p=2, dim=1)),
        'centered_mean_abs_max': centered_mean_abs_max,
        'uncentered_gram_trace': float(saved_uncentered_gram.trace().item()),
        'centered_gram_trace': float(centered_gram.trace().item()),
        'uncentered': _gram_spectrum_diagnostics(saved_uncentered_gram),
        'centered': _gram_spectrum_diagnostics(centered_gram),
        'trace_identity_abs_delta': float(
            (centered_gram.trace() - expected_centered_trace).abs().item()
        ),
        'mean_removal_identity': _delta_stats(
            actual_removed_mean,
            expected_removed_mean,
        ),
        'second_l2_normalization': False,
    }
    return {
        'x_task': x_task,
        'mean': task_mean,
        'centered_features': centered_features,
        'centered_gram': centered_gram,
        'diagnostics': diagnostics,
    }


def build_centered_projection(centered_gram: Tensor, *, scale: float) -> Tensor:
    if float(scale) != EXPECTED_SCALE:
        raise ValueError('E3 requires sap_oracle_scale == 3000')
    projection = build_sap_projection_from_gram(centered_gram, scale=float(scale))
    if not torch.isfinite(projection).all():
        raise ValueError('centered projection contains NaN or Inf')
    torch.testing.assert_close(
        projection,
        projection.transpose(0, 1),
        rtol=0,
        atol=1e-6,
    )
    return projection


def assert_control_task_reconstruction(
    control_weight: Tensor,
    weight_before: Tensor,
    saved_projection: Tensor,
    dataset,
    *,
    last_task_id: int,
) -> dict[str, float | bool]:
    current_start, current_end = map(int, dataset.get_offsets(last_task_id))
    reconstructed, _ = project_linear_weight(
        weight_before[current_start:current_end],
        saved_projection,
    )
    control_current = control_weight[current_start:current_end]
    diagnostics = _delta_stats(reconstructed, control_current)
    if not diagnostics['exact']:
        torch.testing.assert_close(
            reconstructed,
            control_current,
            rtol=1e-6,
            atol=1e-7,
        )
    return diagnostics


def build_centered_candidate(
    control_weight: Tensor,
    weight_before: Tensor,
    centered_projection: Tensor,
    dataset,
    *,
    last_task_id: int,
) -> tuple[Tensor, dict]:
    if control_weight.shape != weight_before.shape:
        raise ValueError('control and W_before classifier shapes differ')
    current_start, current_end = map(int, dataset.get_offsets(last_task_id))
    treatment = control_weight.clone()
    centered_current, projection_stats = project_linear_weight(
        weight_before[current_start:current_end],
        centered_projection,
    )
    treatment[current_start:current_end] = centered_current

    old_mask = torch.ones(
        control_weight.shape[0],
        dtype=torch.bool,
        device=control_weight.device,
    )
    old_mask[current_start:current_end] = False
    old_rows_equal = torch.equal(treatment[old_mask], control_weight[old_mask])
    if not old_rows_equal:
        raise AssertionError('E3 treatment changed old-task classifier rows')
    if not torch.equal(
        treatment[current_start:current_end], centered_current,
    ):
        raise AssertionError('E3 treatment current rows do not match centered projection')

    return treatment, {
        'current_class_offsets': [current_start, current_end],
        'old_rows_bitwise_equal': bool(old_rows_equal),
        **projection_stats,
    }


def save_e3_artifacts(
    output_directory: Path,
    *,
    task_mean: Tensor,
    centered_gram: Tensor,
    centered_projection: Tensor,
    centered_weight: Tensor,
    accuracy: dict,
    gram_diagnostics: dict,
    projection_diagnostics: dict,
    weight_diagnostics: dict,
    manifest: dict,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    tensors = {
        'task10_mean.pt': task_mean,
        'G_task10_centered.pt': centered_gram,
        'M_task10_centered.pt': centered_projection,
        'W_current_centered.pt': centered_weight,
    }
    for filename, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), output_directory / filename)
    payloads = {
        'accuracy.json': accuracy,
        'gram_diagnostics.json': gram_diagnostics,
        'projection_diagnostics.json': projection_diagnostics,
        'weight_diagnostics.json': weight_diagnostics,
        'experiment_manifest.json': manifest,
    }
    for filename, payload in payloads.items():
        (output_directory / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding='utf-8',
        )


def _validate_source_directory(source_artifact_directory: Path) -> None:
    if source_artifact_directory.name != SOURCE_RUN_ID:
        raise ValueError(
            f'E3 source artifact directory must end with run id {SOURCE_RUN_ID}'
        )


def _load_task_ids(path: Path) -> Tensor:
    task_ids = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(task_ids, Tensor) or task_ids.ndim != 1:
        raise TypeError('trusted_task_ids.pt must contain a one-dimensional tensor')
    if task_ids.is_floating_point():
        raise TypeError('trusted_task_ids.pt must use an integer dtype')
    return task_ids.long()


def _projection_diagnostics(
    uncentered_projection: Tensor,
    centered_projection: Tensor,
    *,
    scale: float,
) -> dict:
    symmetry_delta = (
        centered_projection - centered_projection.transpose(0, 1)
    ).abs().max()
    return {
        'sap_oracle_scale': float(scale),
        'uncentered_projection_shape': list(uncentered_projection.shape),
        'centered_projection_shape': list(centered_projection.shape),
        'centered_projection_symmetry_max_abs_delta': float(symmetry_delta.item()),
        'uncentered': _projection_spectrum_diagnostics(uncentered_projection),
        'centered': _projection_spectrum_diagnostics(centered_projection),
        'centered_vs_uncentered': _delta_stats(
            centered_projection,
            uncentered_projection,
        ),
        'importance_formula_reimplemented': False,
    }


def run_e3(
    checkpoint: Path,
    source_artifact_directory: Path,
    output_directory: Path | None = None,
    *,
    device_name: str | None = None,
    sanity_tolerance: float = 1e-4,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    source_artifact_directory = source_artifact_directory.expanduser().resolve()
    _validate_source_directory(source_artifact_directory)
    if output_directory is None:
        output_directory = source_artifact_directory / 'e3_task10_gram_centering'
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f'output directory already exists: {output_directory}')

    required_paths = {
        'checkpoint': checkpoint,
        'W_before': source_artifact_directory / 'W_before.pt',
        'W_after_taskwise': source_artifact_directory / 'W_after_taskwise.pt',
        'X_global': source_artifact_directory / 'X_global.pt',
        'trusted_task_ids': source_artifact_directory / 'trusted_task_ids.pt',
        'G_task_9': source_artifact_directory / 'G_task_9.pt',
        'M_task_9': source_artifact_directory / 'M_task_9.pt',
        'accuracy': source_artifact_directory / 'accuracy.json',
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing E3 source inputs: ' + ', '.join(missing))
    input_files = {
        name: {'path': str(path), 'sha256': e1.sha256_file(path)}
        for name, path in required_paths.items()
    }

    checkpoint_args = mammoth_load_checkpoint(
        str(checkpoint), return_only_args=True,
    )
    provenance = e1.validate_source_provenance(checkpoint_args)
    scale = float(provenance['sap_oracle_scale'])
    if scale != EXPECTED_SCALE:
        raise ValueError('E3 requires source sap_oracle_scale == 3000')
    requested_device = e1._requested_device(device_name)
    if requested_device is not None:
        from utils.conf import get_device
        get_device.device = requested_device
        checkpoint_args.device = None

    model, dataset, checkpoint_args = initialize(checkpoint_args)
    model, _ = mammoth_load_checkpoint(str(checkpoint), model, args=None)
    e1._build_all_test_loaders(dataset)
    last_task_id = validate_task_layout(dataset)
    classifier = resolve_classifier_module(model.net)
    checkpoint_weight = classifier.weight.detach().clone()
    checkpoint_bias = (
        classifier.bias.detach().clone() if classifier.bias is not None else None
    )
    device = checkpoint_weight.device
    dtype = checkpoint_weight.dtype

    weight_before = e1._load_tensor(
        required_paths['W_before'], device=device, dtype=dtype,
    )
    saved_taskwise = e1._load_tensor(
        required_paths['W_after_taskwise'], device=device, dtype=dtype,
    )
    x_global = e1._load_tensor(
        required_paths['X_global'], device=device, dtype=dtype,
    )
    saved_uncentered_gram = e1._load_tensor(
        required_paths['G_task_9'], device=device, dtype=dtype,
    )
    saved_uncentered_projection = e1._load_tensor(
        required_paths['M_task_9'], device=device, dtype=dtype,
    )
    trusted_task_ids = _load_task_ids(required_paths['trusted_task_ids'])

    if tuple(weight_before.shape) != tuple(checkpoint_weight.shape):
        raise ValueError('W_before shape does not match checkpoint classifier')
    if x_global.shape[1] != weight_before.shape[1]:
        raise ValueError('X_global feature dimension does not match classifier input')
    expected_operator_shape = (weight_before.shape[1], weight_before.shape[1])
    if tuple(saved_uncentered_gram.shape) != expected_operator_shape:
        raise ValueError('G_task_9 shape does not match classifier input')
    if tuple(saved_uncentered_projection.shape) != expected_operator_shape:
        raise ValueError('M_task_9 shape does not match classifier input')

    checkpoint_reconstruction = e1._assert_close(
        checkpoint_weight,
        saved_taskwise,
        'checkpoint Current-Local versus W_after_taskwise',
    )
    geometry = build_task10_centered_geometry(
        x_global,
        trusted_task_ids,
        last_task_id=last_task_id,
        saved_uncentered_gram=saved_uncentered_gram,
    )
    centered_projection = build_centered_projection(
        geometry['centered_gram'],
        scale=scale,
    )
    control_reconstruction = assert_control_task_reconstruction(
        checkpoint_weight,
        weight_before,
        saved_uncentered_projection,
        dataset,
        last_task_id=last_task_id,
    )
    centered_weight, candidate_diagnostics = build_centered_candidate(
        checkpoint_weight,
        weight_before,
        centered_projection,
        dataset,
        last_task_id=last_task_id,
    )
    current_start, current_end = candidate_diagnostics['current_class_offsets']
    source_accuracy = json.loads(
        required_paths['accuracy'].read_text(encoding='utf-8')
    )
    if 'taskwise' not in source_accuracy:
        raise ValueError('source accuracy.json is missing taskwise control results')

    projection_diagnostics = _projection_diagnostics(
        saved_uncentered_projection,
        centered_projection,
        scale=scale,
    )
    weight_diagnostics = {
        'current_uncentered_local_vs_w_before': e1.compute_weight_diagnostics(
            weight_before[current_start:current_end],
            checkpoint_weight[current_start:current_end],
        ),
        'current_centered_local_vs_w_before': e1.compute_weight_diagnostics(
            weight_before[current_start:current_end],
            centered_weight[current_start:current_end],
        ),
        'current_centered_vs_uncentered': e1.compute_weight_diagnostics(
            checkpoint_weight[current_start:current_end],
            centered_weight[current_start:current_end],
        ),
        'control_task10_reconstruction': control_reconstruction,
        'old_rows_bitwise_equal': candidate_diagnostics['old_rows_bitwise_equal'],
    }
    sanity_checks = {
        'source_run_id': 'pass',
        'input_provenance': 'pass',
        'task_layout_10_last_id_9': 'pass',
        'checkpoint_matches_source_current_local': 'pass',
        'x10_rows_unit_norm': 'pass',
        'centered_mean_near_zero': 'pass',
        'mean_removal_gram_identity': 'pass',
        'trace_identity': 'pass',
        'control_task10_reconstruction': 'pass',
        'old_rows_bitwise_identical': 'pass',
        'source_current_local_accuracy': 'pending',
    }

    try:
        control_accuracy = e1.evaluate_weight_candidate(
            model,
            dataset,
            classifier,
            checkpoint_weight,
            checkpoint_bias,
        )
        e1._assert_accuracy_matches(
            control_accuracy,
            source_accuracy['taskwise'],
            tolerance=sanity_tolerance,
            description='source Current-Local reconstruction',
        )
        sanity_checks['source_current_local_accuracy'] = 'pass'
        treatment_accuracy = e1.evaluate_weight_candidate(
            model,
            dataset,
            classifier,
            centered_weight,
            checkpoint_bias,
        )
        accuracy = {
            'current_uncentered_local': control_accuracy,
            'current_centered_local': treatment_accuracy,
        }
        manifest = {
            'experiment_name': 'e3_task10_gram_centering',
            'source_run_id': SOURCE_RUN_ID,
            'git_commit': e1._git_commit(),
            'source_checkpoint': str(checkpoint),
            'source_artifact_directory': str(source_artifact_directory),
            'input_files': input_files,
            **provenance,
            'classifier_shape': list(checkpoint_weight.shape),
            'dtype': str(dtype),
            'device': str(device),
            'last_task_id': last_task_id,
            'current_class_offsets': [current_start, current_end],
            'training_performed': False,
            'reference_rebuilt': False,
            'features_reextracted': False,
            'uncentered_gram_recomputed': False,
            'uncentered_projection_recomputed': False,
            'second_l2_normalization': False,
            'inference_feature_centering': False,
            'checkpoint_reconstruction': checkpoint_reconstruction,
            'sanity_checks': sanity_checks,
            'sanity_tolerance_percentage_points': sanity_tolerance,
            'candidate_evaluation_order': [
                'current_uncentered_local',
                'current_centered_local',
            ],
        }
        save_e3_artifacts(
            output_directory,
            task_mean=geometry['mean'],
            centered_gram=geometry['centered_gram'],
            centered_projection=centered_projection,
            centered_weight=centered_weight,
            accuracy=accuracy,
            gram_diagnostics=geometry['diagnostics'],
            projection_diagnostics=projection_diagnostics,
            weight_diagnostics=weight_diagnostics,
            manifest=manifest,
        )
    finally:
        AerSap._install_classifier_candidate(
            classifier,
            checkpoint_weight,
            checkpoint_bias,
        )
    return output_directory


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-artifact-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device')
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    output = run_e3(
        arguments.checkpoint,
        arguments.source_artifact_dir,
        arguments.output_dir,
        device_name=arguments.device,
    )
    print(json.dumps({'status': 'success', 'output_directory': str(output)}))


if __name__ == '__main__':
    main()
