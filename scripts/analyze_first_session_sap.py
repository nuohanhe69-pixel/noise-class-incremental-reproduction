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
# This factor only supplies a floating-point floor for tail-spectrum validation.
# It does not change any experiment, energy definition, or analysis threshold.
DIRECTIONAL_NUMERICAL_FLOOR_SAFETY_FACTOR = 8.0
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
    'common_mean_energy',
    'between_class_energy',
    'within_class_energy',
    'residual_variation_energy',
    'common_mean_fraction',
    'between_class_fraction',
    'within_class_fraction',
    'class_separation_active',
    'class_separation_ratio',
    'between_class_energy_share',
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
DIRECTION_RESULT_KEYS = {
    column: ('eigenvalues' if column == 'eigenvalue' else column)
    for column in DIRECTION_COLUMNS
    if column != 'direction_rank'
}


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


def analyze_feature_centering(
    *,
    x_task1: Tensor,
    raw_gram: Tensor,
    raw_eigenvectors: Tensor,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
    sanity_tolerance: float = 1e-4,
) -> dict:
    """Compare the saved Raw Gram with a mean-centered feature Gram."""
    _require_finite_floating_matrix(x_task1, 'X_task1')
    _require_finite_floating_matrix(raw_gram, 'G_task_0')
    if x_task1.shape[0] == 0:
        raise ValueError('X_task1 must contain at least one sample')
    if raw_gram.shape != (x_task1.shape[1], x_task1.shape[1]):
        raise ValueError('G_task_0 shape does not match X_task1 feature dimension')
    if sanity_tolerance <= 0:
        raise ValueError('sanity_tolerance must be positive')

    device, dtype = _resolve_decomposition_backend(
        raw_gram, decomposition_device, decomposition_dtype,
    )
    if raw_eigenvectors.shape != raw_gram.shape:
        raise ValueError('Raw eigenvector shape does not match G_task_0')
    device_mismatch = (
        raw_eigenvectors.device.type != device.type
        or (
            device.index is not None
            and raw_eigenvectors.device.index != device.index
        )
    )
    if device_mismatch or raw_eigenvectors.dtype != dtype:
        raise ValueError(
            'Raw eigenvectors do not match the requested decomposition backend'
        )
    device = raw_eigenvectors.device
    features = x_task1.detach().to(device=device, dtype=dtype)
    raw_gram_analysis = raw_gram.detach().to(device=device, dtype=dtype)
    sample_count = int(features.shape[0])

    feature_mean = features.mean(dim=0)
    feature_mean_norm_squared = feature_mean.square().sum()
    mean_norm_tolerance = torch.finfo(dtype).eps
    if feature_mean_norm_squared <= mean_norm_tolerance:
        raise ValueError(
            'X_task1 feature mean is too small for cosine analysis: '
            f'norm_squared={feature_mean_norm_squared.item():.6e}'
        )
    mean_energy = sample_count * feature_mean_norm_squared
    raw_trace = raw_gram_analysis.trace()
    if raw_trace <= 0:
        raise ValueError('G_task_0 trace must be positive')
    mean_energy_fraction = mean_energy / raw_trace

    raw_top1 = raw_eigenvectors[:, 0]
    cosine_denominator = raw_top1.square().sum() * feature_mean_norm_squared
    raw_top1_mean_cosine_squared = (
        raw_top1.dot(feature_mean).square() / cosine_denominator
    )

    # X_task1 is already sample-wise L2 normalized by the SAP artifact path.
    # Centering subtracts only the shared feature mean; there is no second L2 step.
    centered_features = features - feature_mean.unsqueeze(0)
    centered_gram = centered_features.T @ centered_features
    expected_centered_gram = (
        raw_gram_analysis - sample_count * torch.outer(feature_mean, feature_mean)
    )
    centered_gram_identity_absolute_error = float(
        (centered_gram - expected_centered_gram).norm().item()
    )
    centered_gram_identity_relative_error = _relative_error(
        centered_gram, expected_centered_gram,
    )
    if centered_gram_identity_relative_error > sanity_tolerance:
        raise ValueError(
            'centered Gram identity failed: '
            f'relative error={centered_gram_identity_relative_error:.6e}'
        )

    centered_trace = centered_gram.trace()
    trace_decomposition_absolute_error = float(
        (raw_trace - (centered_trace + mean_energy)).abs().item()
    )
    trace_decomposition_relative_error = float(
        (trace_decomposition_absolute_error / raw_trace.abs().item())
    )
    if trace_decomposition_relative_error > sanity_tolerance:
        raise ValueError(
            'Gram trace decomposition failed: '
            f'relative error={trace_decomposition_relative_error:.6e}'
        )

    centered_eigenvalues, centered_eigenvectors = decompose_psd_gram(
        centered_gram,
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
    )
    centered_feature_energy_ratio = (
        centered_eigenvalues / centered_eigenvalues.sum()
    )
    centered_cumulative_energy = centered_feature_energy_ratio.cumsum(dim=0)
    positive_centered_ratios = centered_feature_energy_ratio[
        centered_feature_energy_ratio > 0
    ]
    centered_effective_rank = torch.exp(
        -(positive_centered_ratios * positive_centered_ratios.log()).sum(),
    )

    summary = {
        'feature_mean_norm_squared': float(feature_mean_norm_squared.item()),
        'mean_energy': float(mean_energy.item()),
        'mean_energy_fraction': float(mean_energy_fraction.item()),
        'raw_top1_mean_cosine_squared': float(
            raw_top1_mean_cosine_squared.item()
        ),
        'centered_gram_trace': float(centered_trace.item()),
        'centered_top1_feature_energy_ratio': float(
            centered_feature_energy_ratio[:1].sum().item()
        ),
        'centered_top5_feature_energy_ratio': float(
            centered_feature_energy_ratio[:5].sum().item()
        ),
        'centered_top10_feature_energy_ratio': float(
            centered_feature_energy_ratio[:10].sum().item()
        ),
        'centered_top20_feature_energy_ratio': float(
            centered_feature_energy_ratio[:20].sum().item()
        ),
        'centered_top50_feature_energy_ratio': float(
            centered_feature_energy_ratio[:50].sum().item()
        ),
        'centered_effective_rank': float(centered_effective_rank.item()),
        'centered_gram_identity_relative_error': (
            centered_gram_identity_relative_error
        ),
        'centered_gram_identity_absolute_error': (
            centered_gram_identity_absolute_error
        ),
        'trace_decomposition_error': trace_decomposition_relative_error,
        'trace_decomposition_relative_error': trace_decomposition_relative_error,
        'trace_decomposition_absolute_error': trace_decomposition_absolute_error,
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise ValueError('feature-centering diagnostics contain NaN or Inf')

    return {
        'feature_mean': feature_mean,
        'centered_features': centered_features,
        'centered_gram': centered_gram,
        'centered_eigenvalues': centered_eigenvalues,
        'centered_eigenvectors': centered_eigenvectors,
        'centered_feature_energy_ratio': centered_feature_energy_ratio,
        'centered_cumulative_energy': centered_cumulative_energy,
        'summary': summary,
    }


def analyze_raw_direction_class_structure(
    *,
    x_task1: Tensor,
    trusted_labels: Tensor,
    raw_gram: Tensor,
    raw_eigenvalues: Tensor,
    raw_eigenvectors: Tensor,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
    sanity_tolerance: float = 1e-4,
) -> dict:
    """Decompose each Raw Gram direction into common/between/within energy."""
    _require_finite_floating_matrix(x_task1, 'X_task1')
    _require_finite_floating_matrix(raw_gram, 'G_task_0')
    if x_task1.shape[0] == 0:
        raise ValueError('X_task1 must contain at least one sample')
    if trusted_labels.ndim != 1 or len(trusted_labels) != len(x_task1):
        raise ValueError('trusted_labels must align with X_task1 rows')
    if raw_eigenvalues.ndim != 1 or not raw_eigenvalues.is_floating_point():
        raise ValueError('Raw eigenvalues must be a floating-point vector')
    if not torch.isfinite(raw_eigenvalues).all():
        raise ValueError('Raw eigenvalues contain NaN or Inf')
    feature_dimension = int(x_task1.shape[1])
    if raw_gram.shape != (feature_dimension, feature_dimension):
        raise ValueError('G_task_0 does not match X_task1 feature dimension')
    if raw_eigenvalues.shape != (feature_dimension,):
        raise ValueError('Raw eigenvalues do not match X_task1 feature dimension')
    if raw_eigenvectors.shape != (feature_dimension, feature_dimension):
        raise ValueError('Raw eigenvectors do not match X_task1 feature dimension')
    if sanity_tolerance <= 0:
        raise ValueError('sanity_tolerance must be positive')

    requested_device, requested_dtype = _resolve_decomposition_backend(
        x_task1, decomposition_device, decomposition_dtype,
    )
    device_mismatch = (
        raw_eigenvectors.device.type != requested_device.type
        or (
            requested_device.index is not None
            and raw_eigenvectors.device.index != requested_device.index
        )
    )
    if (
        device_mismatch
        or raw_eigenvectors.dtype != requested_dtype
        or raw_eigenvalues.device != raw_eigenvectors.device
        or raw_eigenvalues.dtype != raw_eigenvectors.dtype
    ):
        raise ValueError(
            'Raw eigendecomposition does not match the requested backend'
        )
    analysis_device = raw_eigenvectors.device
    features = x_task1.detach().to(
        device=analysis_device, dtype=requested_dtype,
    )
    raw_gram_analysis = raw_gram.detach().to(
        device=analysis_device, dtype=requested_dtype,
    )
    labels = trusted_labels.detach().to(device=analysis_device).long()
    unique_labels = labels.unique(sorted=True)
    if unique_labels.numel() < 2:
        raise ValueError('class-structure analysis requires at least two classes')

    sample_projections = features @ raw_eigenvectors
    projection_energy = sample_projections.square().sum(dim=0)
    projection_energy_reconstruction_relative_error = _relative_error(
        projection_energy, raw_eigenvalues,
    )
    if projection_energy_reconstruction_relative_error > sanity_tolerance:
        raise ValueError(
            'sample projection energy does not reconstruct Raw eigenvalues: '
            f'relative error={projection_energy_reconstruction_relative_error:.6e}'
        )

    sample_count = int(sample_projections.shape[0])
    global_projection_mean = sample_projections.mean(dim=0)
    common_mean_energy = sample_count * global_projection_mean.square()
    between_class_energy = torch.zeros_like(raw_eigenvalues)
    within_class_energy = torch.zeros_like(raw_eigenvalues)
    for class_label in unique_labels:
        class_projections = sample_projections[labels == class_label]
        class_mean = class_projections.mean(dim=0)
        between_class_energy += class_projections.shape[0] * (
            class_mean - global_projection_mean
        ).square()
        within_class_energy += (
            class_projections - class_mean.unsqueeze(0)
        ).square().sum(dim=0)

    residual_variation_energy = between_class_energy + within_class_energy
    reconstructed_direction_energy = (
        common_mean_energy + residual_variation_energy
    )
    direction_absolute_error = (
        reconstructed_direction_energy - raw_eigenvalues
    ).abs()
    raw_energy_scale = raw_eigenvalues.max()
    if raw_energy_scale <= 0:
        raise ValueError('Raw eigenspectrum has no positive energy')
    raw_trace = raw_gram_analysis.trace()
    numerical_floor = (
        DIRECTIONAL_NUMERICAL_FLOOR_SAFETY_FACTOR
        * torch.finfo(requested_dtype).eps
        * raw_trace.abs()
    )
    stable_energy_threshold = torch.maximum(
        raw_energy_scale * ACTIVE_ENERGY_RELATIVE_THRESHOLD,
        numerical_floor,
    )
    raw_energy_active = (
        raw_eigenvalues > stable_energy_threshold
    )
    direction_relative_error = torch.full_like(raw_eigenvalues, torch.nan)
    positive_energy = raw_eigenvalues > 0
    direction_relative_error[positive_energy] = (
        direction_absolute_error[positive_energy]
        / raw_eigenvalues[positive_energy]
    )
    maximum_direction_relative_error = float(
        direction_relative_error[positive_energy].max().item()
    )
    maximum_stable_direction_relative_error = float(
        direction_relative_error[raw_energy_active].max().item()
    )
    maximum_direction_absolute_error = float(direction_absolute_error.max().item())
    mixed_direction_tolerance = (
        sanity_tolerance * raw_eigenvalues.abs() + numerical_floor
    )
    direction_tolerance_failed = bool(
        (direction_absolute_error > mixed_direction_tolerance).any().item()
    )
    raw_energy_decomposition_absolute_error = float(
        (reconstructed_direction_energy.sum() - raw_trace).abs().item()
    )
    raw_energy_decomposition_relative_error = float(
        raw_energy_decomposition_absolute_error / raw_trace.abs().item()
    )
    if (
        direction_tolerance_failed
        or raw_energy_decomposition_relative_error > sanity_tolerance
    ):
        raise ValueError(
            'Raw directional energy decomposition failed: '
            f'max_direction_relative_error={maximum_direction_relative_error:.6e}, '
            f'max_stable_direction_relative_error='
            f'{maximum_stable_direction_relative_error:.6e}, '
            f'global_relative_error={raw_energy_decomposition_relative_error:.6e}'
        )

    common_mean_fraction = torch.full_like(raw_eigenvalues, torch.nan)
    between_class_fraction = torch.full_like(raw_eigenvalues, torch.nan)
    within_class_fraction = torch.full_like(raw_eigenvalues, torch.nan)
    common_mean_fraction[raw_energy_active] = (
        common_mean_energy[raw_energy_active] / raw_eigenvalues[raw_energy_active]
    )
    between_class_fraction[raw_energy_active] = (
        between_class_energy[raw_energy_active] / raw_eigenvalues[raw_energy_active]
    )
    within_class_fraction[raw_energy_active] = (
        within_class_energy[raw_energy_active] / raw_eigenvalues[raw_energy_active]
    )
    fraction_sum_error = (
        common_mean_fraction[raw_energy_active]
        + between_class_fraction[raw_energy_active]
        + within_class_fraction[raw_energy_active]
        - 1.0
    ).abs()
    maximum_fraction_sum_error = float(fraction_sum_error.max().item())
    relative_stable_threshold = torch.maximum(
        stable_energy_threshold,
        numerical_floor / sanity_tolerance,
    )
    fraction_relative_stable = raw_eigenvalues > relative_stable_threshold
    if fraction_relative_stable.any():
        stable_fraction_sum_error = (
            common_mean_fraction[fraction_relative_stable]
            + between_class_fraction[fraction_relative_stable]
            + within_class_fraction[fraction_relative_stable]
            - 1.0
        ).abs()
        maximum_stable_fraction_sum_error = float(
            stable_fraction_sum_error.max().item()
        )
        stable_fraction_tolerance = (
            sanity_tolerance
            + numerical_floor / raw_eigenvalues[fraction_relative_stable]
        )
        fraction_validation_failed = bool(
            (stable_fraction_sum_error > stable_fraction_tolerance).any().item()
        )
    else:
        maximum_stable_fraction_sum_error = 0.0
        fraction_validation_failed = False
    if fraction_validation_failed:
        raise ValueError(
            'Raw directional energy fractions do not sum to one: '
            f'max_relative_stable_error={maximum_stable_fraction_sum_error:.6e}'
        )

    residual_scale = residual_variation_energy.max()
    stable_residual_threshold = torch.maximum(
        residual_scale * ACTIVE_ENERGY_RELATIVE_THRESHOLD,
        numerical_floor,
    )
    class_separation_active = (
        residual_variation_energy > stable_residual_threshold
    )
    class_separation_ratio = torch.full_like(raw_eigenvalues, torch.nan)
    class_separation_ratio[class_separation_active] = (
        between_class_energy[class_separation_active]
        / residual_variation_energy[class_separation_active]
    )
    common_total = common_mean_energy.sum()
    between_total = between_class_energy.sum()
    within_total = within_class_energy.sum()
    residual_total = between_total + within_total
    if between_total <= torch.finfo(requested_dtype).eps:
        raise ValueError('between-class energy is too small to compute shares')
    if residual_total <= torch.finfo(requested_dtype).eps:
        raise ValueError('residual variation energy is too small')
    between_class_energy_share = between_class_energy / between_total

    summary = {
        'common_mean_energy_total': float(common_total.item()),
        'between_class_energy_total': float(between_total.item()),
        'within_class_energy_total': float(within_total.item()),
        'common_mean_energy_fraction_total': float((common_total / raw_trace).item()),
        'between_class_energy_fraction_total': float((between_total / raw_trace).item()),
        'within_class_energy_fraction_total': float((within_total / raw_trace).item()),
        'global_class_separation_ratio': float((between_total / residual_total).item()),
        'raw_energy_decomposition_relative_error': (
            raw_energy_decomposition_relative_error
        ),
        'raw_energy_decomposition_absolute_error': (
            raw_energy_decomposition_absolute_error
        ),
        'raw_energy_decomposition_max_direction_error': (
            maximum_direction_relative_error
        ),
        'raw_energy_decomposition_max_direction_relative_error': (
            maximum_direction_relative_error
        ),
        'raw_energy_decomposition_max_stable_direction_relative_error': (
            maximum_stable_direction_relative_error
        ),
        'raw_energy_decomposition_max_direction_absolute_error': (
            maximum_direction_absolute_error
        ),
        'raw_energy_decomposition_numerical_floor': float(
            numerical_floor.item()
        ),
        'raw_energy_fraction_sum_max_error': maximum_fraction_sum_error,
        'raw_energy_fraction_sum_max_relative_stable_error': (
            maximum_stable_fraction_sum_error
        ),
        'raw_energy_fraction_relative_stable_count': int(
            fraction_relative_stable.sum().item()
        ),
        'projection_energy_reconstruction_relative_error': (
            projection_energy_reconstruction_relative_error
        ),
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise ValueError('class-structure diagnostics contain NaN or Inf')

    return {
        'sample_projections': sample_projections,
        'common_mean_energy': common_mean_energy,
        'between_class_energy': between_class_energy,
        'within_class_energy': within_class_energy,
        'residual_variation_energy': residual_variation_energy,
        'common_mean_fraction': common_mean_fraction,
        'between_class_fraction': between_class_fraction,
        'within_class_fraction': within_class_fraction,
        'fraction_relative_stable': fraction_relative_stable,
        'class_separation_active': class_separation_active,
        'class_separation_ratio': class_separation_ratio,
        'between_class_energy_share': between_class_energy_share,
        'summary': summary,
    }


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
    result_keys = (
        DIRECTION_RESULT_KEYS[column] for column in DIRECTION_COLUMNS[1:]
    )
    values = [result[key].detach().cpu().tolist() for key in result_keys]
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
            'raw_vs_centered_spectrum.png',
            'Raw vs feature-centered spectrum',
            'Feature energy ratio',
            (
                ('Raw', result['feature_energy_ratio'], '#0072B2', '-'),
                (
                    'Feature-centered', result['centered_feature_energy_ratio'],
                    '#D55E00', '--',
                ),
            ),
        ),
        (
            'raw_vs_centered_cumulative_energy.png',
            'Raw vs feature-centered cumulative energy',
            'Cumulative energy',
            (
                ('Raw', result['cumulative_energy'], '#0072B2', '-'),
                (
                    'Feature-centered', result['centered_cumulative_energy'],
                    '#D55E00', '--',
                ),
            ),
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

    top_count = min(50, result['eigenvalues'].numel())
    top_ranks = torch.arange(1, top_count + 1).numpy()
    common_fraction = result['common_mean_fraction'][:top_count].detach().cpu().numpy()
    between_fraction = result['between_class_fraction'][:top_count].detach().cpu().numpy()
    within_fraction = result['within_class_fraction'][:top_count].detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.bar(
        top_ranks, common_fraction, label='Common mean', color='#0072B2',
        width=0.9,
    )
    axis.bar(
        top_ranks, between_fraction, bottom=common_fraction,
        label='Between-class', color='#D55E00', width=0.9,
    )
    axis.bar(
        top_ranks, within_fraction, bottom=common_fraction + between_fraction,
        label='Within-class', color='#009E73', width=0.9,
    )
    axis.set_xlabel(
        'Raw feature direction rank (descending Raw Gram eigenvalue)'
    )
    axis.set_ylabel('Fraction of raw directional energy')
    axis.set_ylim(0.0, 1.0)
    axis.set_title('Raw direction energy decomposition (Top 50)')
    axis.legend(ncol=3, frameon=False)
    figure.tight_layout()
    decomposition_path = output_directory / 'raw_direction_energy_decomposition.png'
    figure.savefig(decomposition_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    paths.append(decomposition_path)

    active = result['class_separation_active']
    active_shares = result['between_class_energy_share'][active]
    maximum_share = active_shares.max().clamp_min(
        torch.finfo(active_shares.dtype).eps
    )
    point_sizes = 24.0 + 476.0 * torch.sqrt(active_shares / maximum_share)
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.scatter(
        result['sap_importance'][active].detach().cpu().numpy(),
        result['class_separation_ratio'][active].detach().cpu().numpy(),
        s=point_sizes.detach().cpu().numpy(),
        color='#CC79A7', alpha=0.65, edgecolors='none',
    )
    axis.set_xlabel('SAP importance')
    axis.set_ylabel('Class-separation ratio')
    axis.set_title('SAP importance vs Raw-direction class separation')
    axis.set_ylim(-0.02, 1.02)
    figure.tight_layout()
    scatter_path = output_directory / 'sap_importance_vs_discriminativeness.png'
    figure.savefig(scatter_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    paths.append(scatter_path)
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
    centering_result = analyze_feature_centering(
        x_task1=artifacts['x_task1'],
        raw_gram=artifacts['gram'],
        raw_eigenvectors=result['eigenvectors'],
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
        sanity_tolerance=sanity_tolerance,
    )
    result['summary'].update(centering_result.pop('summary'))
    result.update(centering_result)
    class_structure_result = analyze_raw_direction_class_structure(
        x_task1=artifacts['x_task1'],
        trusted_labels=artifacts['trusted_labels'],
        raw_gram=artifacts['gram'],
        raw_eigenvalues=result['eigenvalues'],
        raw_eigenvectors=result['eigenvectors'],
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
        sanity_tolerance=sanity_tolerance,
    )
    result['summary'].update(class_structure_result.pop('summary'))
    result.update(class_structure_result)
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
