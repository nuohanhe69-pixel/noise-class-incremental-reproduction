#!/usr/bin/env python3
"""Offline Task1 Raw/Centered SAP alpha sweep from frozen artifacts."""

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

import scripts.compare_task1_raw_vs_centered_sap as comparison  # noqa: E402
from scripts.analyze_first_session_sap import (  # noqa: E402
    _resolve_decomposition_backend,
)
from utils.sap import (  # noqa: E402
    build_sap_projection_from_gram,
    project_linear_weight,
)


ALPHA_GRID = (1, 10, 30, 100, 300, 1000, 3000, 10000)
REFERENCE_ALPHA = 3000
SUMMARY_COLUMNS = [
    'variant', 'alpha', 'accuracy', 'mean_margin', 'median_margin',
    'prediction_changed_count', 'correct_to_wrong', 'wrong_to_correct',
    'wrong_to_wrong_prediction_changed',
]
PER_CLASS_COLUMNS = [
    'variant', 'alpha', 'class_id', 'sample_count', 'accuracy',
    'accuracy_delta_vs_pre', 'mean_margin', 'mean_margin_delta_vs_pre',
]


def prepare_raw_and_centered_grams(
    reference_features: Tensor,
    *,
    saved_raw_gram: Tensor | None = None,
    decomposition_device: str = 'cuda',
    decomposition_dtype: str = 'float32',
    norm_tolerance: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Prepare the frozen Raw Gram and task-mean-centered Gram on one backend."""
    if reference_features.ndim != 2 or not reference_features.is_floating_point():
        raise ValueError('X_task1 must be a floating-point matrix')
    if not torch.isfinite(reference_features).all():
        raise ValueError('X_task1 contains NaN or Inf')
    row_norms = reference_features.norm(p=2, dim=1)
    torch.testing.assert_close(
        row_norms,
        torch.ones_like(row_norms),
        rtol=norm_tolerance,
        atol=norm_tolerance,
    )
    artifact_gram = (
        reference_features.T @ reference_features
        if saved_raw_gram is None else saved_raw_gram
    )
    if artifact_gram.shape != (
        reference_features.shape[1], reference_features.shape[1],
    ):
        raise ValueError('G_task_0 shape does not match X_task1')
    comparison._assert_close(
        artifact_gram,
        reference_features.T @ reference_features,
        'Raw Gram',
    )
    device, dtype = _resolve_decomposition_backend(
        artifact_gram,
        decomposition_device,
        decomposition_dtype,
    )
    features = reference_features.detach().to(device=device, dtype=dtype)
    raw_gram = artifact_gram.detach().to(device=device, dtype=dtype)
    centered_features = features - features.mean(dim=0, keepdim=True)
    centered_gram = centered_features.T @ centered_features
    if not torch.isfinite(centered_gram).all() or centered_gram.trace() <= 0:
        raise ValueError('Centered Gram must have finite positive energy')
    return raw_gram, centered_gram


def build_candidate(
    weight_before: Tensor,
    gram: Tensor,
    *,
    alpha: float,
    class_count: int = comparison.TASK1_CLASS_COUNT,
) -> dict:
    """Build one seen-row-only candidate with the official SAP projection."""
    if alpha <= 0:
        raise ValueError('SAP alpha must be positive')
    if weight_before.ndim != 2 or gram.shape != (
        weight_before.shape[1], weight_before.shape[1],
    ):
        raise ValueError('classifier and Gram shapes are incompatible')
    if not 0 < class_count <= weight_before.shape[0]:
        raise ValueError('invalid seen-class count')
    projection = build_sap_projection_from_gram(gram, scale=float(alpha))
    weight_backend = weight_before.detach().to(device=gram.device)
    projected_seen, projection_stats = project_linear_weight(
        weight_backend[:class_count],
        projection,
    )
    candidate_backend = weight_backend.clone()
    candidate_backend[:class_count] = projected_seen
    candidate = candidate_backend.to(
        device=weight_before.device,
        dtype=weight_before.dtype,
    )
    if not torch.equal(candidate[class_count:], weight_before[class_count:]):
        raise AssertionError('SAP candidate changed unseen classifier rows')
    return {
        'weight': candidate,
        'projection': projection,
        'projection_stats': projection_stats,
    }


def _relative_frobenius_diff(actual: Tensor, source: Tensor) -> float:
    source = source.to(device=actual.device, dtype=actual.dtype)
    denominator = source.norm().clamp_min(torch.finfo(source.dtype).eps)
    return float(((actual - source).norm() / denominator).item())


def _cross_hardware_drift(
    *,
    recomputed_projection: Tensor,
    source_projection: Tensor,
    recomputed_weight: Tensor,
    source_weight: Tensor,
    recomputed_logits: Tensor,
    source_logits: Tensor,
    recomputed_predictions: Tensor,
    source_predictions: Tensor,
    labels: Tensor,
) -> dict[str, float | int]:
    source_projection_backend = source_projection.to(
        device=recomputed_projection.device,
        dtype=recomputed_projection.dtype,
    )
    projection_delta = recomputed_projection - source_projection_backend
    weight_delta = recomputed_weight - source_weight
    logits_delta = recomputed_logits - source_logits
    recomputed_accuracy = comparison._accuracy(recomputed_predictions, labels)
    source_accuracy = comparison._accuracy(source_predictions, labels)
    return {
        'projection_max_abs_diff': float(projection_delta.abs().max().item()),
        'projection_relative_frobenius_diff': _relative_frobenius_diff(
            recomputed_projection,
            source_projection_backend,
        ),
        'weight_max_abs_diff': float(weight_delta.abs().max().item()),
        'weight_relative_frobenius_diff': _relative_frobenius_diff(
            recomputed_weight,
            source_weight,
        ),
        'logits_mean_abs_diff': float(logits_delta.abs().mean().item()),
        'logits_max_abs_diff': float(logits_delta.abs().max().item()),
        'prediction_disagreement_count': int((
            recomputed_predictions != source_predictions
        ).sum().item()),
        'recomputed_accuracy': recomputed_accuracy,
        'source_accuracy': source_accuracy,
        'accuracy_delta': recomputed_accuracy - source_accuracy,
    }


def _candidate_metrics(
    logits: Tensor,
    labels: Tensor,
    pre_predictions: Tensor,
    pre_margins: Tensor,
) -> tuple[dict, list[dict]]:
    predictions = logits.argmax(dim=1)
    margins = comparison.true_class_margin(logits, labels)
    summary = comparison._candidate_summary(logits, labels, margins)
    summary.update(comparison.paired_candidate_statistics(
        labels,
        pre_predictions,
        predictions,
        margins - pre_margins,
    ))
    per_class = []
    for class_id in range(comparison.TASK1_CLASS_COUNT):
        mask = labels == class_id
        class_predictions = predictions[mask]
        class_labels = labels[mask]
        pre_class_margins = pre_margins[mask]
        class_margins = margins[mask]
        accuracy = comparison._accuracy(class_predictions, class_labels)
        pre_accuracy = comparison._accuracy(pre_predictions[mask], class_labels)
        per_class.append({
            'class_id': class_id,
            'sample_count': int(mask.sum().item()),
            'accuracy': accuracy,
            'accuracy_delta_vs_pre': accuracy - pre_accuracy,
            'mean_margin': float(class_margins.mean().item()),
            'mean_margin_delta_vs_pre': float(
                (class_margins - pre_class_margins).mean().item()
            ),
        })
    return summary, per_class


def _baseline_metrics(
    logits: Tensor,
    labels: Tensor,
) -> tuple[dict, list[dict], Tensor, Tensor]:
    predictions = logits.argmax(dim=1)
    margins = comparison.true_class_margin(logits, labels)
    summary = comparison._candidate_summary(logits, labels, margins)
    summary.update({
        'prediction_changed_count': 0,
        'correct_to_wrong': 0,
        'wrong_to_correct': 0,
        'wrong_to_wrong_prediction_changed': 0,
    })
    per_class = []
    for class_id in range(comparison.TASK1_CLASS_COUNT):
        mask = labels == class_id
        per_class.append({
            'class_id': class_id,
            'sample_count': int(mask.sum().item()),
            'accuracy': comparison._accuracy(predictions[mask], labels[mask]),
            'accuracy_delta_vs_pre': 0.0,
            'mean_margin': float(margins[mask].mean().item()),
            'mean_margin_delta_vs_pre': 0.0,
        })
    return summary, per_class, predictions, margins


def _summary_row(variant: str, alpha: int | str, metrics: dict) -> dict:
    return {
        'variant': variant,
        'alpha': alpha,
        **{column: metrics[column] for column in SUMMARY_COLUMNS[2:]},
    }


def _per_class_rows(
    variant: str,
    alpha: int | str,
    rows: list[dict],
) -> list[dict]:
    return [
        {'variant': variant, 'alpha': alpha, **row}
        for row in rows
    ]


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _save_curve_plot(
    output_path: Path,
    results: list[dict],
    baseline: float,
    metric: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _save_curve_plot_with_pillow(output_path, results, baseline, metric, ylabel)
        return

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.axhline(
        baseline,
        color='#5F6368',
        linestyle='--',
        linewidth=1.6,
        label='Pre-SAP',
    )
    for variant, color, label in (
        ('raw', '#0072B2', 'Raw SAP'),
        ('centered', '#D55E00', 'Centered SAP'),
    ):
        rows = [row for row in results if row['variant'] == variant]
        axis.plot(
            [row['alpha'] for row in rows],
            [row[metric] for row in rows],
            marker='o',
            linewidth=1.8,
            color=color,
            label=label,
        )
    axis.set_xscale('log')
    axis.set_xlabel('SAP alpha (log scale)')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(figure)


def _save_curve_plot_with_pillow(
    output_path: Path,
    results: list[dict],
    baseline: float,
    metric: str,
    ylabel: str,
) -> None:
    """Dependency-light log-alpha curve fallback for test environments."""
    from PIL import Image, ImageDraw

    width, height = 760, 500
    left, top, plot_width, plot_height = 85, 45, 625, 370
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((left, 15), f'Alpha vs {ylabel}', fill='black')
    draw.line((left, top, left, top + plot_height), fill='black', width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill='black', width=2,
    )
    values = [float(row[metric]) for row in results] + [float(baseline)]
    value_min, value_max = min(values), max(values)
    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0
    x_min, x_max = math.log10(ALPHA_GRID[0]), math.log10(ALPHA_GRID[-1])

    def map_x(alpha: float) -> float:
        return left + (math.log10(alpha) - x_min) / (x_max - x_min) * plot_width

    def map_y(value: float) -> float:
        return top + plot_height - (
            (value - value_min) / (value_max - value_min) * plot_height
        )

    baseline_y = map_y(float(baseline))
    draw.line(
        (left, baseline_y, left + plot_width, baseline_y),
        fill='#5F6368', width=2,
    )
    for variant, color in (('raw', '#0072B2'), ('centered', '#D55E00')):
        rows = [row for row in results if row['variant'] == variant]
        points = [(map_x(row['alpha']), map_y(float(row[metric]))) for row in rows]
        draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((left + 5, height - 38), 'SAP alpha (log scale)', fill='black')
    draw.text((8, top + plot_height // 2), ylabel, fill='black')
    image.save(output_path)


def run_sweep(
    artifact_directory: Path,
    output_directory: Path | None = None,
    *,
    decomposition_device: str = 'cuda',
    decomposition_dtype: str = 'float32',
    sanity_tolerance: float = 1e-5,
) -> Path:
    """Run the fixed Raw/Centered alpha grid without loading or training a model."""
    artifact_directory = artifact_directory.expanduser().resolve()
    if output_directory is None:
        output_directory = artifact_directory / 'task1_sap_alpha_sweep'
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f'output directory already exists: {output_directory}')

    paths = {
        name: artifact_directory / filename
        for name, filename in comparison.REQUIRED_TENSORS.items()
    }
    paths['manifest'] = artifact_directory / 'manifest.json'
    paths['decision_stats'] = artifact_directory / 'task1_test_decision_stats.json'
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing Task1 alpha-sweep inputs: ' + ', '.join(missing))
    tensors = {
        name: comparison._load_tensor(path)
        for name, path in paths.items()
        if name not in {'manifest', 'decision_stats'}
    }
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    decision_stats = json.loads(paths['decision_stats'].read_text(encoding='utf-8'))
    bias_path = artifact_directory / 'classifier_bias_before.pt'
    bias = None
    if bool(manifest.get('classifier_has_bias', False)):
        if not bias_path.is_file():
            raise FileNotFoundError(f'missing classifier bias: {bias_path}')
        bias = comparison._load_tensor(bias_path)
    bias_before = None if bias is None else bias.clone()
    if manifest.get('test_feature_type') != 'raw_classifier_input':
        raise ValueError('alpha sweep requires raw classifier-input test features')
    if manifest.get('test_feature_l2_normalized') is not False:
        raise ValueError('Task1 test feature normalization metadata is invalid')

    reference = tensors['x_task1']
    features = tensors['test_features']
    labels = tensors['test_labels'].long()
    weight_before = tensors['weight_before']
    weight_after = tensors['weight_after']
    if (
        features.ndim != 2
        or labels.ndim != 1
        or features.shape[0] != labels.shape[0]
        or features.shape[1] != weight_before.shape[1]
    ):
        raise ValueError('Task1 test features, labels, and classifier do not align')
    if labels.unique(sorted=True).tolist() != list(range(comparison.TASK1_CLASS_COUNT)):
        raise ValueError('Task1 alpha sweep requires exactly classes 0-9')

    raw_gram, centered_gram = prepare_raw_and_centered_grams(
        reference,
        saved_raw_gram=tensors['raw_gram'],
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
    )
    pre_logits_full = torch.nn.functional.linear(features, weight_before, bias)
    comparison._assert_close(
        pre_logits_full, tensors['pre_logits'], 'Pre-SAP logits',
    )
    pre_logits = pre_logits_full[:, :comparison.TASK1_CLASS_COUNT]
    pre_predictions_saved = tensors['pre_predictions'].long()
    if not torch.equal(pre_logits.argmax(dim=1), pre_predictions_saved):
        raise AssertionError('Pre-SAP saved predictions do not match reconstruction')
    if abs(
        comparison._accuracy(pre_predictions_saved, labels)
        - float(decision_stats['pre_accuracy_evaluator'])
    ) > sanity_tolerance:
        raise AssertionError('Pre-SAP evaluator accuracy does not match reconstruction')
    baseline, baseline_per_class, pre_predictions, pre_margins = _baseline_metrics(
        pre_logits,
        labels,
    )

    source_weight_reconstructed = weight_before.clone()
    try:
        source_weight_reconstructed[
            :comparison.TASK1_CLASS_COUNT
        ], _ = project_linear_weight(
            weight_before[:comparison.TASK1_CLASS_COUNT],
            tensors['raw_projection'],
        )
        torch.testing.assert_close(
            source_weight_reconstructed,
            weight_after,
            rtol=1e-5,
            atol=1e-6,
        )
    except AssertionError as error:
        raise AssertionError(
            'source alpha=3000 Raw artifact is internally inconsistent: '
            'saved M_task_0 + W_before does not reconstruct W_after'
        ) from error

    source_raw_logits_full = torch.nn.functional.linear(
        features,
        weight_after,
        bias,
    )
    try:
        torch.testing.assert_close(
            source_raw_logits_full,
            tensors['raw_logits'],
            rtol=1e-5,
            atol=1e-6,
        )
    except AssertionError as error:
        raise AssertionError(
            'source alpha=3000 Raw artifact is internally inconsistent: '
            'saved W_after does not reconstruct saved post-SAP logits'
        ) from error
    source_raw_predictions = source_raw_logits_full[
        :, :comparison.TASK1_CLASS_COUNT
    ].argmax(dim=1)
    if not torch.equal(source_raw_predictions, tensors['raw_predictions'].long()):
        raise AssertionError(
            'source alpha=3000 Raw artifact is internally inconsistent: '
            'saved post-SAP predictions do not match saved logits'
        )
    source_raw_accuracy = comparison._accuracy(source_raw_predictions, labels)
    if abs(
        source_raw_accuracy
        - float(decision_stats['post_accuracy_evaluator'])
    ) > sanity_tolerance:
        raise AssertionError(
            'source alpha=3000 Raw artifact is internally inconsistent: '
            'saved post-SAP accuracy does not match evaluator accuracy'
        )
    source_raw_metrics, _ = _candidate_metrics(
        source_raw_logits_full[:, :comparison.TASK1_CLASS_COUNT],
        labels,
        pre_predictions,
        pre_margins,
    )

    raw_reference = build_candidate(
        weight_before,
        raw_gram,
        alpha=REFERENCE_ALPHA,
    )
    recomputed_raw_logits_full = torch.nn.functional.linear(
        features,
        raw_reference['weight'],
        bias,
    )
    recomputed_raw_predictions = recomputed_raw_logits_full[
        :, :comparison.TASK1_CLASS_COUNT
    ].argmax(dim=1)
    recomputed_raw_metrics, _ = _candidate_metrics(
        recomputed_raw_logits_full[:, :comparison.TASK1_CLASS_COUNT],
        labels,
        pre_predictions,
        pre_margins,
    )
    cross_hardware_drift = _cross_hardware_drift(
        recomputed_projection=raw_reference['projection'],
        source_projection=tensors['raw_projection'],
        recomputed_weight=raw_reference['weight'],
        source_weight=weight_after,
        recomputed_logits=recomputed_raw_logits_full,
        source_logits=tensors['raw_logits'],
        recomputed_predictions=recomputed_raw_predictions,
        source_predictions=tensors['raw_predictions'].long(),
        labels=labels,
    )

    candidate_cache = {('raw', REFERENCE_ALPHA): raw_reference}
    result_rows = []
    class_rows = []
    unseen_rows_unchanged = True
    for alpha in ALPHA_GRID:
        for variant, gram in (('raw', raw_gram), ('centered', centered_gram)):
            candidate = candidate_cache.get((variant, alpha))
            if candidate is None:
                candidate = build_candidate(
                    weight_before,
                    gram,
                    alpha=alpha,
                )
            unseen_rows_unchanged = unseen_rows_unchanged and torch.equal(
                candidate['weight'][comparison.TASK1_CLASS_COUNT:],
                weight_before[comparison.TASK1_CLASS_COUNT:],
            )
            logits = torch.nn.functional.linear(
                features,
                candidate['weight'],
                bias,
            )[:, :comparison.TASK1_CLASS_COUNT]
            metrics, per_class = _candidate_metrics(
                logits,
                labels,
                pre_predictions,
                pre_margins,
            )
            result_rows.append({'variant': variant, 'alpha': alpha, **metrics})
            class_rows.extend(_per_class_rows(variant, alpha, per_class))
    if not unseen_rows_unchanged:
        raise AssertionError('one or more sweep candidates changed unseen rows')
    bias_unchanged = bias is None or torch.equal(bias, bias_before)
    if not bias_unchanged:
        raise AssertionError('classifier bias changed during alpha sweep')

    summary_csv_rows = [_summary_row('pre_sap', '', baseline)] + [
        _summary_row(row['variant'], row['alpha'], row)
        for row in result_rows
    ]
    per_class_csv_rows = _per_class_rows('pre_sap', '', baseline_per_class) + class_rows
    payload = {
        'experiment_name': 'task1_raw_centered_sap_alpha_sweep',
        'alpha_grid': list(ALPHA_GRID),
        'baseline': baseline,
        'baseline_per_class': baseline_per_class,
        'results': result_rows,
        'per_class_results': class_rows,
        'sample_count': int(labels.numel()),
        'classes': list(range(comparison.TASK1_CLASS_COUNT)),
        'decomposition_device': str(raw_gram.device),
        'decomposition_dtype': str(raw_gram.dtype).removeprefix('torch.'),
        'second_l2_normalization': False,
        'training_performed': False,
        'source_raw_alpha3000': {
            'metrics': source_raw_metrics,
            'internal_consistency': True,
        },
        'recomputed_raw_alpha3000': {
            'metrics': recomputed_raw_metrics,
            'cross_hardware_drift': cross_hardware_drift,
        },
        'sanity_checks': {
            'pre_sap_reconstruction': True,
            'source_raw_alpha3000_internal_consistency': True,
            'recomputed_raw_alpha3000_drift_recorded': True,
            'bias_unchanged': bool(bias_unchanged),
            'unseen_rows_unchanged': bool(unseen_rows_unchanged),
        },
        'source_artifact_paths': {
            name: str(path.resolve()) for name, path in paths.items()
        },
    }
    if bias is not None:
        payload['source_artifact_paths']['classifier_bias'] = str(bias_path.resolve())

    output_directory.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output_directory / 'task1_sap_alpha_sweep_summary.csv',
        SUMMARY_COLUMNS,
        summary_csv_rows,
    )
    _write_csv(
        output_directory / 'task1_sap_alpha_sweep_per_class.csv',
        PER_CLASS_COLUMNS,
        per_class_csv_rows,
    )
    (output_directory / 'task1_sap_alpha_sweep_summary.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8',
    )
    _save_curve_plot(
        output_directory / '01_alpha_vs_accuracy.png',
        result_rows,
        baseline['accuracy'],
        'accuracy',
        'Task1 accuracy (%)',
    )
    _save_curve_plot(
        output_directory / '02_alpha_vs_mean_margin.png',
        result_rows,
        baseline['mean_margin'],
        'mean_margin',
        'Task1 mean true-class margin',
    )
    return output_directory


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--decomposition-device', default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument(
        '--decomposition-dtype', default='float32',
        choices=('artifact', 'float32', 'float64'),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    output = run_sweep(
        arguments.artifact_dir,
        arguments.output_dir,
        decomposition_device=arguments.decomposition_device,
        decomposition_dtype=arguments.decomposition_dtype,
    )
    print(json.dumps({'status': 'success', 'output_directory': str(output)}))


if __name__ == '__main__':
    main()
