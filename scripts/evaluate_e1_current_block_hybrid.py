#!/usr/bin/env python3
"""Offline E1 evaluation for the final task's classifier row block."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from main import initialize  # noqa: E402
from models.aer_sap import AerSap  # noqa: E402
from utils.checkpoints import mammoth_load_checkpoint  # noqa: E402
from utils.sap import project_linear_weight, resolve_classifier_module  # noqa: E402


ACCURACY_FIELDS = (
    'class_il',
    'task_il',
    'per_task_class_il',
    'per_task_task_il',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_provenance(args) -> dict:
    """Fail unless checkpoint arguments identify the reviewed source run."""
    provenance = {
        'dataset': getattr(args, 'dataset', None),
        'model': getattr(args, 'model', None),
        'backbone': getattr(args, 'backbone', None),
        'seed': getattr(args, 'seed', None),
        'noise_rate': getattr(args, 'noise_rate', None),
        'noise_type': getattr(args, 'noise_type', None),
        'sap_oracle_scale': getattr(args, 'sap_oracle_scale', None),
    }
    failures = []
    if provenance['dataset'] != 'seq-cifar100':
        failures.append('dataset must be seq-cifar100')
    if str(provenance['model']).replace('_', '-') != 'aer-sap':
        failures.append('model must be aer-sap')
    if provenance['seed'] != 0:
        failures.append('seed must be 0')
    if provenance['noise_type'] not in {'symm', 'symmetric'}:
        failures.append('noise_type must be symm or symmetric')
    if provenance['noise_rate'] is None or abs(float(provenance['noise_rate']) - 0.2) > 1e-12:
        failures.append('noise_rate must be 0.2')
    if provenance['sap_oracle_scale'] is None or abs(
        float(provenance['sap_oracle_scale']) - 3000.0
    ) > 1e-12:
        failures.append('sap_oracle_scale must be 3000')
    if int(getattr(args, 'debug_mode', 0) or 0) != 0:
        failures.append('debug_mode must be disabled')
    if bool(getattr(args, 'eval_future', False)):
        failures.append('eval_future must be disabled')
    if failures:
        raise ValueError('source checkpoint provenance mismatch: ' + '; '.join(failures))
    return provenance


def _load_tensor(path: Path, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    tensor = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(tensor, Tensor):
        raise TypeError(f'{path} does not contain a tensor')
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f'{path} must contain a finite floating-point tensor')
    return tensor.to(device=device, dtype=dtype)


def _assert_close(actual: Tensor, expected: Tensor, description: str) -> dict:
    if actual.shape != expected.shape:
        raise AssertionError(
            f'{description} shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}'
        )
    difference = (actual - expected).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    denominator = expected.norm().clamp_min(torch.finfo(expected.dtype).eps)
    relative = float((actual - expected).norm().div(denominator).item())
    exact = torch.equal(actual, expected)
    if not exact:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    return {
        'exact': bool(exact),
        'max_abs_delta': maximum,
        'relative_delta': relative,
    }


def build_hybrid_candidates(
    weight_before: Tensor,
    task_matrices: dict[int, Tensor],
    global_matrix: Tensor,
    dataset,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Build one shared old-local base, then replace only the current rows."""
    task_count = int(dataset.N_TASKS)
    last_task_id = task_count - 1
    offsets = [tuple(map(int, dataset.get_offsets(task_id))) for task_id in range(task_count)]

    old_local_base = weight_before.clone()
    for task_id in range(last_task_id):
        start_class, end_class = offsets[task_id]
        projected, _ = project_linear_weight(
            weight_before[start_class:end_class], task_matrices[task_id],
        )
        old_local_base[start_class:end_class] = projected

    current_start, current_end = offsets[last_task_id]
    current_weight = weight_before[current_start:current_end]
    current_local, _ = project_linear_weight(
        current_weight, task_matrices[last_task_id],
    )
    current_global, _ = project_linear_weight(current_weight, global_matrix)

    candidates = {
        'current_local': old_local_base.clone(),
        'current_identity': old_local_base.clone(),
        'current_global': old_local_base.clone(),
    }
    candidates['current_local'][current_start:current_end] = current_local
    candidates['current_global'][current_start:current_end] = current_global
    return old_local_base, candidates


def assert_candidate_row_contracts(
    candidates: dict[str, Tensor],
    weight_before: Tensor,
    dataset,
) -> tuple[int, int]:
    last_task_id = int(dataset.N_TASKS) - 1
    current_start, current_end = map(int, dataset.get_offsets(last_task_id))
    old_mask = torch.ones(weight_before.shape[0], dtype=torch.bool, device=weight_before.device)
    old_mask[current_start:current_end] = False
    local = candidates['current_local']
    identity = candidates['current_identity']
    global_candidate = candidates['current_global']
    if not torch.equal(local[old_mask], identity[old_mask]):
        raise AssertionError('Current-Local and Current-Identity old rows differ')
    if not torch.equal(local[old_mask], global_candidate[old_mask]):
        raise AssertionError('Current-Local and Current-Global old rows differ')
    if not torch.equal(
        identity[current_start:current_end],
        weight_before[current_start:current_end],
    ):
        raise AssertionError('Current-Identity current rows differ from W_before')
    return current_start, current_end


def compute_weight_diagnostics(weight_before: Tensor, weight_after: Tensor) -> dict[str, float]:
    if torch.equal(weight_before, weight_after):
        return {
            'relative_weight_delta': 0.0,
            'weight_norm_ratio': 1.0,
            'cosine': 1.0,
        }
    denominator = weight_before.norm().clamp_min(torch.finfo(weight_before.dtype).eps)
    cosine = torch.nn.functional.cosine_similarity(
        weight_before.flatten(), weight_after.flatten(), dim=0,
    )
    return {
        'relative_weight_delta': float(
            ((weight_after - weight_before).norm() / denominator).item()
        ),
        'weight_norm_ratio': float((weight_after.norm() / denominator).item()),
        'cosine': float(cosine.item()),
    }


def assert_identity_diagnostics(diagnostics: dict[str, float]) -> None:
    expected = {
        'relative_weight_delta': 0.0,
        'weight_norm_ratio': 1.0,
        'cosine': 1.0,
    }
    for key, value in expected.items():
        if diagnostics[key] != value:
            raise AssertionError(f'identity {key} is {diagnostics[key]}, expected {value}')


def _assert_accuracy_matches(
    actual: dict,
    expected: dict,
    *,
    tolerance: float,
    description: str,
) -> None:
    for field in ACCURACY_FIELDS:
        if field not in actual or field not in expected:
            raise AssertionError(f'{description} is missing {field}')
        actual_values = actual[field] if isinstance(actual[field], list) else [actual[field]]
        expected_values = expected[field] if isinstance(expected[field], list) else [expected[field]]
        if len(actual_values) != len(expected_values):
            raise AssertionError(f'{description} {field} length mismatch')
        for index, (actual_value, expected_value) in enumerate(
            zip(actual_values, expected_values)
        ):
            if abs(float(actual_value) - float(expected_value)) > tolerance:
                raise AssertionError(
                    f'{description} {field}[{index}] differs: '
                    f'{actual_value} != {expected_value}'
                )


def evaluate_identity_sanity(
    weight_before: Tensor,
    expected_accuracy: dict,
    evaluator: Callable[[Tensor], dict],
    *,
    tolerance: float,
) -> dict:
    actual = evaluator(weight_before)
    _assert_accuracy_matches(
        actual,
        expected_accuracy,
        tolerance=tolerance,
        description='identity reconstruction',
    )
    return actual


def assert_post_evaluation_contracts(
    identity_accuracy: dict,
    candidate_accuracy: dict[str, dict],
    *,
    last_task_id: int,
    tolerance: float,
) -> None:
    task_il = {
        name: result['per_task_task_il']
        for name, result in candidate_accuracy.items()
    }
    for task_id in range(last_task_id):
        baseline = float(task_il['current_local'][task_id])
        for name in ('current_identity', 'current_global'):
            if abs(float(task_il[name][task_id]) - baseline) > tolerance:
                raise AssertionError(f'{name} old Task-IL differs at task {task_id}')
    identity_current = float(identity_accuracy['per_task_task_il'][last_task_id])
    hybrid_identity_current = float(task_il['current_identity'][last_task_id])
    if abs(hybrid_identity_current - identity_current) > tolerance:
        raise AssertionError('Current-Identity current Task-IL differs from Identity')


def evaluate_weight_candidate(
    model,
    dataset,
    classifier,
    candidate_weight: Tensor,
    checkpoint_bias: Tensor | None,
) -> dict:
    AerSap._install_classifier_candidate(
        classifier, candidate_weight, checkpoint_bias,
    )
    return AerSap._summarize_evaluation(dataset, model)


def save_e1_artifacts(
    output_directory: Path,
    candidates: dict[str, Tensor],
    accuracy: dict,
    diagnostics: dict,
    manifest: dict,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    filenames = {
        'current_local': 'W_current_local.pt',
        'current_identity': 'W_current_identity.pt',
        'current_global': 'W_current_global.pt',
    }
    for candidate_name, filename in filenames.items():
        torch.save(
            candidates[candidate_name].detach().cpu(),
            output_directory / filename,
        )
    for filename, payload in (
        ('accuracy.json', accuracy),
        ('weight_diagnostics.json', diagnostics),
        ('experiment_manifest.json', manifest),
    ):
        (output_directory / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8',
        )


def _requested_device(device_name: str | None) -> torch.device | None:
    if device_name is None:
        return None
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested but is unavailable')
    return device


def _git_commit() -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _build_all_test_loaders(dataset) -> None:
    for _ in range(int(dataset.N_TASKS)):
        dataset.get_data_loaders()
    if len(dataset.test_loaders) != int(dataset.N_TASKS):
        raise AssertionError('failed to build every task test loader')


def run_e1(
    checkpoint: Path,
    source_artifact_directory: Path,
    output_directory: Path | None = None,
    *,
    device_name: str | None = None,
    sanity_tolerance: float = 1e-4,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    source_artifact_directory = source_artifact_directory.expanduser().resolve()
    if output_directory is None:
        output_directory = source_artifact_directory / 'e1_current_block_hybrid'
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f'output directory already exists: {output_directory}')

    checkpoint_args = mammoth_load_checkpoint(
        str(checkpoint), return_only_args=True,
    )
    provenance = validate_source_provenance(checkpoint_args)
    requested_device = _requested_device(device_name)
    if requested_device is not None:
        from utils.conf import get_device
        get_device.device = requested_device
        checkpoint_args.device = None

    model, dataset, checkpoint_args = initialize(checkpoint_args)
    model, _ = mammoth_load_checkpoint(str(checkpoint), model, args=None)
    _build_all_test_loaders(dataset)
    classifier = resolve_classifier_module(model.net)
    checkpoint_weight = classifier.weight.detach().clone()
    checkpoint_bias = (
        classifier.bias.detach().clone() if classifier.bias is not None else None
    )

    required_paths = {
        'checkpoint': checkpoint,
        'W_before': source_artifact_directory / 'W_before.pt',
        'W_after_taskwise': source_artifact_directory / 'W_after_taskwise.pt',
        'M_global': source_artifact_directory / 'M_global.pt',
        'accuracy': source_artifact_directory / 'accuracy.json',
    }
    for task_id in range(int(dataset.N_TASKS)):
        required_paths[f'M_task_{task_id}'] = (
            source_artifact_directory / f'M_task_{task_id}.pt'
        )
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing E1 source inputs: ' + ', '.join(missing))
    input_files = {
        name: {'path': str(path), 'sha256': sha256_file(path)}
        for name, path in required_paths.items()
    }

    dtype = checkpoint_weight.dtype
    device = checkpoint_weight.device
    weight_before = _load_tensor(required_paths['W_before'], device=device, dtype=dtype)
    saved_taskwise = _load_tensor(
        required_paths['W_after_taskwise'], device=device, dtype=dtype,
    )
    global_matrix = _load_tensor(required_paths['M_global'], device=device, dtype=dtype)
    task_matrices = {
        task_id: _load_tensor(
            required_paths[f'M_task_{task_id}'], device=device, dtype=dtype,
        )
        for task_id in range(int(dataset.N_TASKS))
    }
    if tuple(weight_before.shape) != tuple(checkpoint_weight.shape):
        raise ValueError('W_before shape does not match the checkpoint classifier')
    expected_matrix_shape = (weight_before.shape[1], weight_before.shape[1])
    for name, matrix in [('M_global', global_matrix), *[
        (f'M_task_{task_id}', matrix) for task_id, matrix in task_matrices.items()
    ]]:
        if tuple(matrix.shape) != expected_matrix_shape:
            raise ValueError(f'{name} shape does not match classifier input dimension')

    source_accuracy = json.loads(required_paths['accuracy'].read_text(encoding='utf-8'))
    if 'identity' not in source_accuracy:
        raise ValueError('source accuracy.json is missing identity results')

    sanity_checks = {
        'check_0_input_provenance': 'pass',
        'check_1_checkpoint_matches_taskwise': 'pending',
        'check_2_identity_reconstruction': 'pending',
        'check_3_current_local_reconstruction': 'pending',
        'check_4_old_rows_identical': 'pending',
        'check_5_identity_current_rows': 'pending',
        'check_6_old_task_il_identical': 'pending',
        'check_7_identity_current_task_il': 'pending',
    }
    try:
        checkpoint_reconstruction = _assert_close(
            checkpoint_weight, saved_taskwise,
            'checkpoint classifier versus W_after_taskwise',
        )
        sanity_checks['check_1_checkpoint_matches_taskwise'] = 'pass'

        evaluator = lambda weight: evaluate_weight_candidate(
            model, dataset, classifier, weight, checkpoint_bias,
        )
        identity_accuracy = evaluate_identity_sanity(
            weight_before,
            source_accuracy['identity'],
            evaluator,
            tolerance=sanity_tolerance,
        )
        sanity_checks['check_2_identity_reconstruction'] = 'pass'

        _old_local_base, candidates = build_hybrid_candidates(
            weight_before, task_matrices, global_matrix, dataset,
        )
        local_reconstruction = _assert_close(
            candidates['current_local'], saved_taskwise,
            'Current-Local versus W_after_taskwise',
        )
        sanity_checks['check_3_current_local_reconstruction'] = 'pass'

        current_start, current_end = assert_candidate_row_contracts(
            candidates, weight_before, dataset,
        )
        sanity_checks['check_4_old_rows_identical'] = 'pass'
        sanity_checks['check_5_identity_current_rows'] = 'pass'

        candidate_accuracy = {
            name: evaluator(candidates[name])
            for name in ('current_local', 'current_identity', 'current_global')
        }
        last_task_id = int(dataset.N_TASKS) - 1
        assert_post_evaluation_contracts(
            identity_accuracy,
            candidate_accuracy,
            last_task_id=last_task_id,
            tolerance=sanity_tolerance,
        )
        sanity_checks['check_6_old_task_il_identical'] = 'pass'
        sanity_checks['check_7_identity_current_task_il'] = 'pass'

        diagnostics = {
            name: compute_weight_diagnostics(
                weight_before[current_start:current_end],
                candidate[current_start:current_end],
            )
            for name, candidate in candidates.items()
        }
        assert_identity_diagnostics(diagnostics['current_identity'])
        accuracy = {'identity_sanity': identity_accuracy, **candidate_accuracy}
        manifest = {
            'experiment_name': 'e1_current_block_hybrid',
            'git_commit': _git_commit(),
            'source_checkpoint': str(checkpoint),
            'source_artifact_directory': str(source_artifact_directory),
            'input_files': input_files,
            **provenance,
            'classifier_shape': list(checkpoint_weight.shape),
            'dtype': str(checkpoint_weight.dtype),
            'device': str(checkpoint_weight.device),
            'last_task_id': last_task_id,
            'current_class_offsets': [current_start, current_end],
            'training_performed': False,
            'reference_rebuilt': False,
            'features_reextracted': False,
            'gram_recomputed': False,
            'projection_matrices_recomputed': False,
            'sanity_checks': sanity_checks,
            'sanity_tolerance_percentage_points': sanity_tolerance,
            'checkpoint_reconstruction': checkpoint_reconstruction,
            'current_local_reconstruction': local_reconstruction,
            'candidate_evaluation_order': [
                'identity_sanity',
                'current_local',
                'current_identity',
                'current_global',
            ],
        }
        save_e1_artifacts(
            output_directory, candidates, accuracy, diagnostics, manifest,
        )
    finally:
        AerSap._install_classifier_candidate(
            classifier, checkpoint_weight, checkpoint_bias,
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
    output = run_e1(
        arguments.checkpoint,
        arguments.source_artifact_dir,
        arguments.output_dir,
        device_name=arguments.device,
    )
    print(json.dumps({'status': 'success', 'output_directory': str(output)}))


if __name__ == '__main__':
    main()
