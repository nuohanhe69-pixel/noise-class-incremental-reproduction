#!/usr/bin/env python3
"""Offline spectrum and classifier-energy diagnostics for Task1 SAP artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.sap import _sap_importance  # noqa: E402


FEATURE_DIMENSION = 512
TASK1_CLASS_COUNT = 10
TOTAL_CLASS_COUNT = 100
ACTIVE_ENERGY_RELATIVE_THRESHOLD = 1e-12
DECOMPOSITION_DTYPES = {
    'float32': torch.float32,
    'float64': torch.float64,
}
TENSOR_FILENAMES = {
    'x_task1': 'X_task1.pt',
    'gram': 'G_task_0.pt',
    'saved_projection': 'M_task_0.pt',
    'weight_before': 'W_before.pt',
    'weight_after': 'W_after.pt',
    'trusted_labels': 'trusted_labels.pt',
    'trusted_task_ids': 'trusted_task_ids.pt',
}
JSON_FILENAMES = {
    'manifest': 'manifest.json',
    'reference_stats': 'reference_stats.json',
    'coverage': 'coverage.json',
    'accuracy': 'accuracy.json',
}
DIRECTION_COLUMNS = (
    'direction_rank',
    'eigenvalue',
    'feature_energy_ratio',
    'cumulative_energy',
    'sap_importance',
    'classifier_energy_before',
    'classifier_energy_after',
    'classifier_energy_active',
    'classifier_energy_ratio',
    'classifier_energy_loss',
    'classifier_energy_loss_ratio',
    'centered_classifier_energy_before',
    'centered_classifier_energy_after',
    'centered_classifier_energy_active',
    'centered_classifier_energy_ratio',
    'centered_classifier_energy_loss',
    'centered_classifier_energy_loss_ratio',
)


def _require_finite_floating_matrix(value: Tensor, name: str) -> None:
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating-point matrix')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} contains NaN or Inf')


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    if actual.shape != expected.shape:
        raise ValueError(
            f'relative-error shape mismatch: {tuple(actual.shape)} != '
            f'{tuple(expected.shape)}'
        )
    denominator = expected.norm().clamp_min(torch.finfo(expected.dtype).eps)
    return float(((actual - expected).norm() / denominator).item())


def _energy_change_diagnostics(
    energy_before: Tensor,
    energy_after: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return active mask, ratio, loss, and loss ratio for directional energy."""
    if energy_before.shape != energy_after.shape:
        raise ValueError('before/after energy shapes differ')
    active_threshold = (
        energy_before.max() * ACTIVE_ENERGY_RELATIVE_THRESHOLD
    )
    active = energy_before > active_threshold
    ratio = torch.full_like(energy_before, torch.nan)
    ratio[active] = energy_after[active] / energy_before[active]
    loss = energy_before - energy_after
    loss_ratio = torch.full_like(energy_before, torch.nan)
    loss_ratio[active] = 1.0 - ratio[active]
    return active, ratio, loss, loss_ratio


def _resolve_decomposition_backend(
    gram: Tensor,
    decomposition_device: str,
    decomposition_dtype: str,
) -> tuple[torch.device, torch.dtype]:
    device = torch.device(decomposition_device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('decomposition_device must be cpu or cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA decomposition requested but CUDA is unavailable')
    if decomposition_dtype == 'artifact':
        dtype = gram.dtype
    else:
        try:
            dtype = DECOMPOSITION_DTYPES[decomposition_dtype]
        except KeyError as error:
            raise ValueError(
                'decomposition_dtype must be artifact, float32, or float64'
            ) from error
    if dtype not in DECOMPOSITION_DTYPES.values():
        raise ValueError('decomposition requires float32 or float64')
    return device, dtype


def decompose_psd_gram(
    gram: Tensor,
    *,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
) -> tuple[Tensor, Tensor]:
    """Return non-negative eigenvalues/eigenvectors in descending-energy order."""
    _require_finite_floating_matrix(gram, 'G_task_0')
    if gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise ValueError(f'G_task_0 must be square, got shape {tuple(gram.shape)}')

    device, dtype = _resolve_decomposition_backend(
        gram, decomposition_device, decomposition_dtype,
    )
    decomposition_gram = gram.detach().to(device=device, dtype=dtype)
    symmetric_gram = (decomposition_gram + decomposition_gram.T) * 0.5
    asymmetry_ratio = _relative_error(decomposition_gram, symmetric_gram)
    if asymmetry_ratio > 1e-5:
        raise ValueError(
            f'G_task_0 is not numerically symmetric: relative error={asymmetry_ratio:.3e}'
        )

    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_gram)
    spectral_scale = max(float(eigenvalues.abs().max().item()), 1.0)
    relative_negative_tolerance = (
        1e-5 if dtype == torch.float32
        else 1e-10
    )
    negative_tolerance = relative_negative_tolerance * spectral_scale
    minimum_eigenvalue = float(eigenvalues.min().item())
    if minimum_eigenvalue < -negative_tolerance:
        raise ValueError(
            'G_task_0 has an obviously negative eigenvalue: '
            f'min={minimum_eigenvalue:.6e}, tolerance={negative_tolerance:.6e}'
        )

    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    eigenvectors = eigenvectors.flip(1)
    if eigenvalues.sum() <= 0:
        raise ValueError('G_task_0 has no positive feature energy')
    return eigenvalues, eigenvectors


def analyze_task1_geometry(
    *,
    gram: Tensor,
    saved_projection: Tensor,
    weight_before: Tensor,
    weight_after: Tensor,
    alpha: float,
    task1_row_count: int = TASK1_CLASS_COUNT,
    sanity_tolerance: float = 1e-4,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
) -> dict:
    """Analyze one saved Task1 SAP geometry without changing any artifact."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    if sanity_tolerance <= 0:
        raise ValueError('sanity_tolerance must be positive')

    eigenvalues, eigenvectors = decompose_psd_gram(
        gram,
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
    )
    feature_dimension = int(eigenvalues.numel())
    for tensor, name in (
        (saved_projection, 'M_task_0'),
        (weight_before, 'W_before'),
        (weight_after, 'W_after'),
    ):
        _require_finite_floating_matrix(tensor, name)
    if saved_projection.shape != (feature_dimension, feature_dimension):
        raise ValueError('M_task_0 shape does not match G_task_0')
    if weight_before.shape != weight_after.shape:
        raise ValueError('W_before and W_after shapes differ')
    if weight_before.shape[1] != feature_dimension:
        raise ValueError('classifier input dimension does not match G_task_0')
    if not 0 < task1_row_count <= weight_before.shape[0]:
        raise ValueError('invalid Task1 classifier row count')
    if not torch.equal(
        weight_before[task1_row_count:], weight_after[task1_row_count:],
    ):
        raise ValueError('classifier rows after Task1 changed')

    energy_total = eigenvalues.sum()
    feature_energy_ratio = eigenvalues / energy_total
    cumulative_energy = feature_energy_ratio.cumsum(dim=0)
    # Reuse the exact SAP scaling helper used by the training implementation:
    # m_j = alpha*r_j / ((alpha - 1)*r_j + 1).
    importance = _sap_importance(eigenvalues, float(alpha))
    reconstructed_projection = (
        eigenvectors * importance.unsqueeze(0)
    ) @ eigenvectors.T
    reconstructed_projection = (
        reconstructed_projection + reconstructed_projection.T
    ) * 0.5

    analysis_device = eigenvalues.device
    analysis_dtype = eigenvalues.dtype
    saved_projection_analysis = saved_projection.detach().to(
        device=analysis_device, dtype=analysis_dtype,
    )
    m_reconstruction_error = _relative_error(
        reconstructed_projection, saved_projection_analysis,
    )
    if m_reconstruction_error > sanity_tolerance:
        raise ValueError(
            'saved projection is inconsistent with the current SAP formula: '
            f'relative error={m_reconstruction_error:.6e}'
        )

    weight_before_analysis = weight_before.detach().to(
        device=analysis_device, dtype=analysis_dtype,
    )
    weight_after_analysis = weight_after.detach().to(
        device=analysis_device, dtype=analysis_dtype,
    )
    expected_task1_weight = (
        weight_before_analysis[:task1_row_count] @ saved_projection_analysis.T
    )
    projection_weight_error = _relative_error(
        weight_after_analysis[:task1_row_count], expected_task1_weight,
    )
    if projection_weight_error > sanity_tolerance:
        raise ValueError(
            'Task1 classifier block is inconsistent with W_before @ M_task_0.T: '
            f'relative error={projection_weight_error:.6e}'
        )

    task_weight_before = weight_before_analysis[:task1_row_count]
    task_weight_after = weight_after_analysis[:task1_row_count]
    coefficients_before = task_weight_before @ eigenvectors
    coefficients_after = task_weight_after @ eigenvectors
    classifier_energy_before = coefficients_before.square().sum(dim=0)
    classifier_energy_after = coefficients_after.square().sum(dim=0)
    theoretical_energy_after = importance.square() * classifier_energy_before
    theoretical_energy_error = _relative_error(
        classifier_energy_after, theoretical_energy_after,
    )
    if theoretical_energy_error > sanity_tolerance:
        raise ValueError(
            'classifier energy does not satisfy E_after = m^2 * E_before: '
            f'relative error={theoretical_energy_error:.6e}'
        )

    centered_weight_before = task_weight_before - task_weight_before.mean(
        dim=0, keepdim=True,
    )
    centered_weight_after = task_weight_after - task_weight_after.mean(
        dim=0, keepdim=True,
    )
    centered_coefficients_before = centered_weight_before @ eigenvectors
    centered_coefficients_after = centered_weight_after @ eigenvectors
    centered_classifier_energy_before = centered_coefficients_before.square().sum(dim=0)
    centered_classifier_energy_after = centered_coefficients_after.square().sum(dim=0)
    centered_theoretical_energy_after = (
        importance.square() * centered_classifier_energy_before
    )
    centered_theoretical_energy_error = _relative_error(
        centered_classifier_energy_after, centered_theoretical_energy_after,
    )
    if centered_theoretical_energy_error > sanity_tolerance:
        raise ValueError(
            'centered classifier energy does not satisfy '
            'E_after = m^2 * E_before: '
            f'relative error={centered_theoretical_energy_error:.6e}'
        )

    (
        classifier_energy_active,
        classifier_energy_ratio,
        classifier_energy_loss,
        classifier_energy_loss_ratio,
    ) = _energy_change_diagnostics(
        classifier_energy_before, classifier_energy_after,
    )
    (
        centered_classifier_energy_active,
        centered_classifier_energy_ratio,
        centered_classifier_energy_loss,
        centered_classifier_energy_loss_ratio,
    ) = _energy_change_diagnostics(
        centered_classifier_energy_before, centered_classifier_energy_after,
    )
    epsilon = torch.finfo(analysis_dtype).eps
    positive_ratios = feature_energy_ratio[feature_energy_ratio > 0]
    # Standard effective rank: exp(H(p)), where H(p) = -sum_j p_j log(p_j)
    # and p is the normalized non-negative Gram eigenspectrum.
    effective_rank = torch.exp(
        -(positive_ratios * positive_ratios.log()).sum(),
    )
    energy_before_total = classifier_energy_before.sum()
    energy_after_total = classifier_energy_after.sum()
    centered_energy_before_total = centered_classifier_energy_before.sum()
    centered_energy_after_total = centered_classifier_energy_after.sum()

    summary = {
        'alpha': float(alpha),
        'decomposition_device': str(analysis_device),
        'decomposition_dtype': str(analysis_dtype).removeprefix('torch.'),
        'gram_trace': float(gram.detach().cpu().trace().item()),
        'top1_feature_energy_ratio': float(feature_energy_ratio[:1].sum().item()),
        'top5_feature_energy_ratio': float(feature_energy_ratio[:5].sum().item()),
        'top10_feature_energy_ratio': float(feature_energy_ratio[:10].sum().item()),
        'top20_feature_energy_ratio': float(feature_energy_ratio[:20].sum().item()),
        'top50_feature_energy_ratio': float(feature_energy_ratio[:50].sum().item()),
        'effective_rank': float(effective_rank.item()),
        'classifier_energy_before_total': float(energy_before_total.item()),
        'classifier_energy_after_total': float(energy_after_total.item()),
        'classifier_energy_total_ratio': float(
            (energy_after_total / energy_before_total.clamp_min(epsilon)).item()
        ),
        'centered_classifier_energy_before_total': float(
            centered_energy_before_total.item()
        ),
        'centered_classifier_energy_after_total': float(
            centered_energy_after_total.item()
        ),
        'centered_classifier_energy_total_ratio': float(
            (
                centered_energy_after_total
                / centered_energy_before_total.clamp_min(epsilon)
            ).item()
        ),
        'm_reconstruction_relative_error': m_reconstruction_error,
        'w_projection_reconstruction_relative_error': projection_weight_error,
        'theoretical_energy_relation_error': theoretical_energy_error,
        'centered_theoretical_energy_relation_error': (
            centered_theoretical_energy_error
        ),
        'importance_gt_0_9_count': int((importance > 0.9).sum().item()),
        'importance_gt_0_5_count': int((importance > 0.5).sum().item()),
        'importance_lt_0_1_count': int((importance < 0.1).sum().item()),
    }
    numeric_summary_values = (
        value for value in summary.values() if isinstance(value, (int, float))
    )
    if not all(math.isfinite(value) for value in numeric_summary_values):
        raise ValueError('diagnostic summary contains NaN or Inf')

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'feature_energy_ratio': feature_energy_ratio,
        'cumulative_energy': cumulative_energy,
        'sap_importance': importance,
        'classifier_energy_before': classifier_energy_before,
        'classifier_energy_after': classifier_energy_after,
        'classifier_energy_active': classifier_energy_active,
        'classifier_energy_ratio': classifier_energy_ratio,
        'classifier_energy_loss': classifier_energy_loss,
        'classifier_energy_loss_ratio': classifier_energy_loss_ratio,
        'centered_classifier_energy_before': centered_classifier_energy_before,
        'centered_classifier_energy_after': centered_classifier_energy_after,
        'centered_classifier_energy_active': centered_classifier_energy_active,
        'centered_classifier_energy_ratio': centered_classifier_energy_ratio,
        'centered_classifier_energy_loss': centered_classifier_energy_loss,
        'centered_classifier_energy_loss_ratio': (
            centered_classifier_energy_loss_ratio
        ),
        'summary': summary,
    }


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value


def _resolve_alpha(manifest: dict, alpha_override: float | None) -> float:
    manifest_alpha = manifest.get('sap_alpha')
    if manifest_alpha is None:
        if alpha_override is None:
            raise ValueError('manifest has no sap_alpha; provide --alpha')
        alpha = float(alpha_override)
    else:
        alpha = float(manifest_alpha)
        if alpha_override is not None and not math.isclose(
            float(alpha_override), alpha, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError('--alpha does not match manifest sap_alpha')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('artifact alpha must be finite and positive')
    return alpha


def load_first_session_artifacts(
    artifact_directory: Path,
    *,
    alpha_override: float | None = None,
) -> dict:
    """Load and validate the exact artifacts emitted by ``AerSap`` Task1."""
    artifact_directory = artifact_directory.expanduser().resolve()
    if not artifact_directory.is_dir():
        raise NotADirectoryError(artifact_directory)

    loaded = {}
    for key, filename in TENSOR_FILENAMES.items():
        path = artifact_directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        value = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(value, Tensor):
            raise TypeError(f'{filename} does not contain a tensor')
        loaded[key] = value
    for key, filename in JSON_FILENAMES.items():
        loaded[key] = _load_json(artifact_directory / filename)

    x_task1 = loaded['x_task1']
    gram = loaded['gram']
    projection = loaded['saved_projection']
    weight_before = loaded['weight_before']
    weight_after = loaded['weight_after']
    if x_task1.ndim != 2 or tuple(x_task1.shape[1:]) != (FEATURE_DIMENSION,):
        raise ValueError(f'X_task1 must have shape [N, {FEATURE_DIMENSION}]')
    if not x_task1.is_floating_point() or not torch.isfinite(x_task1).all():
        raise ValueError('X_task1 must be a finite floating-point matrix')
    if tuple(gram.shape) != (FEATURE_DIMENSION, FEATURE_DIMENSION):
        raise ValueError('G_task_0 must have shape [512, 512]')
    if tuple(projection.shape) != (FEATURE_DIMENSION, FEATURE_DIMENSION):
        raise ValueError('M_task_0 must have shape [512, 512]')
    expected_weight_shape = (TOTAL_CLASS_COUNT, FEATURE_DIMENSION)
    if tuple(weight_before.shape) != expected_weight_shape:
        raise ValueError('W_before must have shape [100, 512]')
    if tuple(weight_after.shape) != expected_weight_shape:
        raise ValueError('W_after must have shape [100, 512]')
    reference_count = int(x_task1.shape[0])
    for key in ('trusted_labels', 'trusted_task_ids'):
        tensor = loaded[key]
        if tensor.ndim != 1 or len(tensor) != reference_count:
            raise ValueError(f'{TENSOR_FILENAMES[key]} must align with X_task1 rows')
    if not torch.equal(
        loaded['trusted_task_ids'].long(),
        torch.zeros(reference_count, dtype=torch.long),
    ):
        raise ValueError('Task1 artifacts must contain only trusted task id 0')

    loaded['alpha'] = _resolve_alpha(loaded['manifest'], alpha_override)
    loaded['artifact_directory'] = artifact_directory
    return loaded


def _direction_rows(result: dict):
    tensor_keys = DIRECTION_COLUMNS[1:]
    values = [result[key].detach().cpu().tolist() for key in tensor_keys]
    for index, direction_values in enumerate(zip(*values), start=1):
        yield (index, *direction_values)


def _write_direction_summary(result: dict, output_directory: Path) -> Path:
    path = output_directory / 'direction_summary.csv'
    with path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(DIRECTION_COLUMNS)
        writer.writerows(_direction_rows(result))
    return path


def _plot_diagnostics(result: dict, output_directory: Path, dpi: int) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            'plotting requires matplotlib; install the project optional dependencies'
        ) from error

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
    })
    ranks = torch.arange(1, result['eigenvalues'].numel() + 1).numpy()
    plot_specs = (
        (
            'feature_spectrum.png', 'Feature energy spectrum',
            'Feature energy ratio',
            (('Feature energy', result['feature_energy_ratio'], '#0072B2', '-'),),
        ),
        (
            'cumulative_energy.png', 'Cumulative feature energy',
            'Cumulative energy',
            (('Cumulative energy', result['cumulative_energy'], '#009E73', '-'),),
        ),
        (
            'sap_importance.png', 'SAP importance', 'SAP importance',
            (('Importance', result['sap_importance'], '#E69F00', '-'),),
        ),
        (
            'classifier_energy_before_after.png', 'Classifier energy by feature direction',
            'Classifier energy',
            (
                ('Before SAP', result['classifier_energy_before'], '#0072B2', '-'),
                ('After SAP', result['classifier_energy_after'], '#D55E00', '--'),
            ),
        ),
        (
            'classifier_energy_ratio.png', 'Classifier energy retained by SAP',
            'E_after / E_before',
            (('Energy ratio', result['classifier_energy_ratio'], '#CC79A7', '-'),),
        ),
        (
            'centered_classifier_energy_before_after.png',
            'Centered classifier energy by feature direction',
            'Centered classifier energy',
            (
                (
                    'Before SAP', result['centered_classifier_energy_before'],
                    '#0072B2', '-',
                ),
                (
                    'After SAP', result['centered_classifier_energy_after'],
                    '#D55E00', '--',
                ),
            ),
        ),
    )
    paths = []
    for filename, title, ylabel, series in plot_specs:
        figure, axis = plt.subplots(figsize=(7.2, 4.0))
        for label, tensor, color, linestyle in series:
            axis.plot(
                ranks, tensor.detach().cpu().numpy(), label=label,
                color=color, linestyle=linestyle, linewidth=1.8,
            )
        axis.set_xlabel('Feature direction rank (descending Gram eigenvalue)')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
        if len(series) > 1:
            axis.legend(frameon=False)
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(figure)
        paths.append(path)
    return paths


def save_diagnostics(result: dict, output_directory: Path, *, dpi: int = 300) -> list[Path]:
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_paths = _plot_diagnostics(result, output_directory, dpi)
    csv_path = _write_direction_summary(result, output_directory)
    summary_path = output_directory / 'diagnostic_summary.json'
    summary_path.write_text(
        json.dumps(result['summary'], indent=2, sort_keys=True), encoding='utf-8',
    )
    return [*plot_paths, csv_path, summary_path]


def run_diagnostic(
    artifact_directory: Path,
    output_directory: Path,
    *,
    alpha_override: float | None = None,
    sanity_tolerance: float = 1e-4,
    dpi: int = 300,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
) -> list[Path]:
    artifacts = load_first_session_artifacts(
        artifact_directory, alpha_override=alpha_override,
    )
    result = analyze_task1_geometry(
        gram=artifacts['gram'],
        saved_projection=artifacts['saved_projection'],
        weight_before=artifacts['weight_before'],
        weight_after=artifacts['weight_after'],
        alpha=artifacts['alpha'],
        task1_row_count=TASK1_CLASS_COUNT,
        sanity_tolerance=sanity_tolerance,
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
    )
    print(f"decomposition_device={result['summary']['decomposition_device']}")
    print(f"decomposition_dtype={result['summary']['decomposition_dtype']}")
    return save_diagnostics(result, output_directory, dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument(
        '--alpha', type=float, default=None,
        help='Required only when manifest.json does not contain sap_alpha.',
    )
    parser.add_argument('--sanity_tolerance', type=float, default=1e-4)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument(
        '--decomposition_device', type=str, default='cpu',
        help='Eigendecomposition device: cpu, cuda, or an explicit CUDA index.',
    )
    parser.add_argument(
        '--decomposition_dtype', type=str, default='artifact',
        choices=('artifact', 'float32', 'float64'),
        help='Eigendecomposition dtype; artifact preserves G_task_0 dtype.',
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.dpi <= 0:
        raise ValueError('--dpi must be positive')
    output_paths = run_diagnostic(
        arguments.artifact_dir,
        arguments.output_dir,
        alpha_override=arguments.alpha,
        sanity_tolerance=arguments.sanity_tolerance,
        dpi=arguments.dpi,
        decomposition_device=arguments.decomposition_device,
        decomposition_dtype=arguments.decomposition_dtype,
    )
    for path in output_paths:
        print(path)


if __name__ == '__main__':
    main()
