#!/usr/bin/env python3
"""Analyze Task1 Class-4 decisions under Pre, Raw, and Centered SAP."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.compare_task1_raw_vs_centered_sap as comparison  # noqa: E402
from utils.sap import project_linear_weight  # noqa: E402


TARGET_CLASS = 4
CLASS_3 = 3
CLASS_2 = 2
SAMPLE_DETAIL_COLUMNS = [
    'index', 'label',
    'pre_prediction', 'raw_prediction', 'centered_prediction',
    'pre_margin', 'raw_margin', 'centered_margin',
    'raw_delta_margin', 'centered_delta_margin',
    'pre_competitor', 'raw_competitor', 'centered_competitor',
    'pre_gap_4_vs_3', 'raw_gap_4_vs_3', 'centered_gap_4_vs_3',
    'pre_gap_4_vs_2', 'raw_gap_4_vs_2', 'centered_gap_4_vs_2',
]


def _strongest_competitor(logits: Tensor, target_class: int) -> Tensor:
    competitors = logits.clone()
    competitors[:, target_class] = -torch.inf
    return competitors.argmax(dim=1)


def _frequency(values: Tensor) -> dict[str, int]:
    unique, counts = values.unique(sorted=True, return_counts=True)
    return {
        str(int(value.item())): int(count.item())
        for value, count in zip(unique, counts)
    }


def _class_candidate_summary(
    logits: Tensor,
    labels: Tensor,
    margins: Tensor,
    competitors: Tensor,
) -> dict:
    predictions = logits.argmax(dim=1)
    return {
        'accuracy': comparison._accuracy(predictions, labels),
        'mean_margin': float(margins.mean().item()),
        'median_margin': float(margins.median().item()),
        'strongest_competitor_frequency': _frequency(competitors),
        'mean_gap_4_vs_3': float(
            (logits[:, TARGET_CLASS] - logits[:, CLASS_3]).mean().item()
        ),
        'mean_gap_4_vs_2': float(
            (logits[:, TARGET_CLASS] - logits[:, CLASS_2]).mean().item()
        ),
    }


def analyze_class4_decisions(
    labels: Tensor,
    pre_logits: Tensor,
    raw_logits: Tensor,
    centered_logits: Tensor,
) -> dict:
    """Return Class-4 sample metrics from aligned seen-class logits."""
    if labels.ndim != 1:
        raise ValueError('Task1 test labels must be one-dimensional')
    if not (
        pre_logits.shape == raw_logits.shape == centered_logits.shape
        and pre_logits.ndim == 2
        and pre_logits.shape[0] == labels.shape[0]
        and pre_logits.shape[1] == comparison.TASK1_CLASS_COUNT
    ):
        raise ValueError('Pre/Raw/Centered Task1 logits must align and cover classes 0-9')
    sample_indices = torch.where(labels == TARGET_CLASS)[0]
    if sample_indices.numel() == 0:
        raise ValueError('Task1 test artifacts contain no Class-4 samples')

    class_labels = labels[sample_indices]
    pre = pre_logits[sample_indices]
    raw = raw_logits[sample_indices]
    centered = centered_logits[sample_indices]
    pre_predictions = pre.argmax(dim=1)
    raw_predictions = raw.argmax(dim=1)
    centered_predictions = centered.argmax(dim=1)
    pre_margin = comparison.true_class_margin(pre, class_labels)
    raw_margin = comparison.true_class_margin(raw, class_labels)
    centered_margin = comparison.true_class_margin(centered, class_labels)
    raw_delta = raw_margin - pre_margin
    centered_delta = centered_margin - pre_margin
    pre_competitor = _strongest_competitor(pre, TARGET_CLASS)
    raw_competitor = _strongest_competitor(raw, TARGET_CLASS)
    centered_competitor = _strongest_competitor(centered, TARGET_CLASS)

    result = {
        'sample_indices': sample_indices,
        'labels': class_labels,
        'pre_prediction': pre_predictions,
        'raw_prediction': raw_predictions,
        'centered_prediction': centered_predictions,
        'pre_margin': pre_margin,
        'raw_margin': raw_margin,
        'centered_margin': centered_margin,
        'raw_delta_margin': raw_delta,
        'centered_delta_margin': centered_delta,
        'pre_competitor': pre_competitor,
        'raw_competitor': raw_competitor,
        'centered_competitor': centered_competitor,
        'pre_gap_4_vs_3': pre[:, TARGET_CLASS] - pre[:, CLASS_3],
        'raw_gap_4_vs_3': raw[:, TARGET_CLASS] - raw[:, CLASS_3],
        'centered_gap_4_vs_3': centered[:, TARGET_CLASS] - centered[:, CLASS_3],
        'pre_gap_4_vs_2': pre[:, TARGET_CLASS] - pre[:, CLASS_2],
        'raw_gap_4_vs_2': raw[:, TARGET_CLASS] - raw[:, CLASS_2],
        'centered_gap_4_vs_2': centered[:, TARGET_CLASS] - centered[:, CLASS_2],
    }
    result['summary'] = {
        'target_class': TARGET_CLASS,
        'class4_sample_count': int(sample_indices.numel()),
        'pre_sap': _class_candidate_summary(
            pre, class_labels, pre_margin, pre_competitor,
        ),
        'raw_sap': _class_candidate_summary(
            raw, class_labels, raw_margin, raw_competitor,
        ),
        'centered_sap': _class_candidate_summary(
            centered, class_labels, centered_margin, centered_competitor,
        ),
        'raw_vs_pre': comparison.paired_candidate_statistics(
            class_labels, pre_predictions, raw_predictions, raw_delta,
        ),
        'centered_vs_pre': comparison.paired_candidate_statistics(
            class_labels, pre_predictions, centered_predictions, centered_delta,
        ),
    }
    return result


def _resolve_existing_summary(
    artifact_directory: Path,
    comparison_summary_path: Path | None,
) -> Path:
    if comparison_summary_path is not None:
        path = comparison_summary_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'comparison summary does not exist: {path}')
        return path
    candidates = (
        artifact_directory / 'task1_raw_vs_centered_sap'
        / 'raw_vs_centered_summary.json',
        artifact_directory / 'raw_vs_centered_summary.json',
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'raw_vs_centered_summary.json is required for overall sanity checks'
    )


def _assert_overall_summary(
    actual: dict,
    expected: dict,
    *,
    tolerance: float,
) -> None:
    for candidate in ('pre_sap', 'raw_sap', 'centered_sap'):
        for metric in ('accuracy', 'mean_margin', 'median_margin'):
            difference = abs(
                float(actual[candidate][metric])
                - float(expected[candidate][metric])
            )
            if difference > tolerance:
                raise AssertionError(
                    'existing Raw-vs-Centered summary mismatch: '
                    f'{candidate}.{metric} absolute_error={difference:.6e}'
                )


def _detail_rows(result: dict) -> list[dict]:
    rows = []
    for row_index in range(result['sample_indices'].numel()):
        row = {}
        for column in SAMPLE_DETAIL_COLUMNS:
            key = 'sample_indices' if column == 'index' else (
                'labels' if column == 'label' else column
            )
            value = result[key][row_index].item()
            row[column] = int(value) if column in {
                'index', 'label', 'pre_prediction', 'raw_prediction',
                'centered_prediction', 'pre_competitor', 'raw_competitor',
                'centered_competitor',
            } else float(value)
        rows.append(row)
    return rows


def _save_figure(output_path: Path, result: dict) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _save_figure_with_pillow(output_path, result)
        return

    order = torch.argsort(result['pre_margin']).cpu().numpy()
    x = torch.arange(len(order)).numpy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'Pre-SAP': '#7A7A7A', 'Raw SAP': '#0072B2', 'Centered SAP': '#D55E00'}
    for label, key in (
        ('Pre-SAP', 'pre_margin'),
        ('Raw SAP', 'raw_margin'),
        ('Centered SAP', 'centered_margin'),
    ):
        axes[0].plot(
            x, result[key].detach().cpu().numpy()[order],
            label=label, color=colors[label], linewidth=1.6,
        )
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set(
        title='A. Class 4 margins (Pre-SAP order)',
        xlabel='Class 4 test sample rank', ylabel='True-class margin',
    )
    axes[0].legend(frameon=False)

    positions = torch.arange(2, dtype=torch.float32).numpy()
    width = 0.24
    gap_keys = ('gap_4_vs_3', 'gap_4_vs_2')
    for offset, (label, prefix) in zip(
        (-width, 0.0, width),
        (('Pre-SAP', 'pre'), ('Raw SAP', 'raw'), ('Centered SAP', 'centered')),
    ):
        values = [
            float(result[f'{prefix}_{gap_key}'].mean().item())
            for gap_key in gap_keys
        ]
        axes[1].bar(
            positions + offset, values, width=width,
            label=label, color=colors[label],
        )
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set(
        title='B. Mean Class 4 logit gaps',
        ylabel='Mean logit gap',
        xticks=positions,
        xticklabels=('Class 4 vs 3', 'Class 4 vs 2'),
    )
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _save_figure_with_pillow(output_path: Path, result: dict) -> None:
    """Dependency-light two-panel PNG fallback for test environments."""
    from PIL import Image, ImageDraw

    width, height = 1300, 500
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    colors = {'pre': '#7A7A7A', 'raw': '#0072B2', 'centered': '#D55E00'}

    order = torch.argsort(result['pre_margin'])
    margins = torch.cat((
        result['pre_margin'], result['raw_margin'], result['centered_margin'],
    )).float()
    margin_min, margin_max = float(margins.min()), float(margins.max())
    if margin_min == margin_max:
        margin_min -= 1.0
        margin_max += 1.0
    left, top, panel_width, panel_height = 70, 50, 520, 380
    draw.text((left, 15), 'A. Class 4 margins (Pre-SAP order)', fill='black')
    draw.line((left, top, left, top + panel_height), fill='black', width=2)
    draw.line((left, top + panel_height, left + panel_width, top + panel_height), fill='black', width=2)

    def margin_y(value: float) -> float:
        return top + panel_height - (value - margin_min) / (margin_max - margin_min) * panel_height

    zero_y = margin_y(0.0)
    draw.line((left, zero_y, left + panel_width, zero_y), fill='black', width=1)
    count = max(int(order.numel()) - 1, 1)
    for key, color in colors.items():
        values = result[f'{key}_margin'][order].tolist()
        points = [
            (left + index / count * panel_width, margin_y(float(value)))
            for index, value in enumerate(values)
        ]
        if len(points) == 1:
            draw.ellipse((points[0][0] - 2, points[0][1] - 2, points[0][0] + 2, points[0][1] + 2), fill=color)
        else:
            draw.line(points, fill=color, width=3)

    panel_left = 720
    draw.text((panel_left, 15), 'B. Mean Class 4 logit gaps', fill='black')
    gap_values = {
        prefix: [
            float(result[f'{prefix}_gap_4_vs_3'].mean()),
            float(result[f'{prefix}_gap_4_vs_2'].mean()),
        ]
        for prefix in ('pre', 'raw', 'centered')
    }
    all_gaps = [value for values in gap_values.values() for value in values] + [0.0]
    gap_min, gap_max = min(all_gaps), max(all_gaps)
    if gap_min == gap_max:
        gap_min -= 1.0
        gap_max += 1.0

    def gap_y(value: float) -> float:
        return top + panel_height - (value - gap_min) / (gap_max - gap_min) * panel_height

    draw.line((panel_left, top, panel_left, top + panel_height), fill='black', width=2)
    draw.line((panel_left, top + panel_height, 1240, top + panel_height), fill='black', width=2)
    gap_zero = gap_y(0.0)
    draw.line((panel_left, gap_zero, 1240, gap_zero), fill='black', width=1)
    for group_index, gap_name in enumerate(('Class 4 vs 3', 'Class 4 vs 2')):
        center = panel_left + 150 + group_index * 250
        for method_index, prefix in enumerate(('pre', 'raw', 'centered')):
            value = gap_values[prefix][group_index]
            x0 = center + (method_index - 1) * 45
            draw.rectangle(
                (x0, min(gap_zero, gap_y(value)), x0 + 34, max(gap_zero, gap_y(value))),
                fill=colors[prefix],
            )
        draw.text((center - 35, top + panel_height + 12), gap_name, fill='black')
    image.save(output_path)


def run_class4_analysis(
    artifact_directory: Path,
    output_directory: Path | None = None,
    *,
    comparison_summary_path: Path | None = None,
    decomposition_device: str | None = None,
    decomposition_dtype: str | None = None,
    sanity_tolerance: float = 1e-5,
) -> Path:
    artifact_directory = artifact_directory.expanduser().resolve()
    existing_summary_path = _resolve_existing_summary(
        artifact_directory,
        comparison_summary_path,
    )
    existing_summary = json.loads(existing_summary_path.read_text(encoding='utf-8'))
    decomposition_device = (
        decomposition_device or existing_summary['decomposition_device']
    )
    decomposition_dtype = (
        decomposition_dtype or existing_summary['decomposition_dtype']
    )
    if output_directory is None:
        output_directory = artifact_directory / 'task1_class4_decision_analysis'
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
        raise FileNotFoundError('missing Task1 Class-4 inputs: ' + ', '.join(missing))
    tensors = {
        name: comparison._load_tensor(path)
        for name, path in paths.items()
        if name not in {'manifest', 'decision_stats'}
    }
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    decision_stats = json.loads(paths['decision_stats'].read_text(encoding='utf-8'))
    bias = None
    if bool(manifest.get('classifier_has_bias', False)):
        bias_path = artifact_directory / 'classifier_bias_before.pt'
        if not bias_path.is_file():
            raise FileNotFoundError(f'missing classifier bias: {bias_path}')
        bias = comparison._load_tensor(bias_path)
    if manifest.get('test_feature_type') != 'raw_classifier_input':
        raise ValueError('Task1 test features are not raw classifier inputs')
    if manifest.get('test_feature_l2_normalized') is not False:
        raise ValueError('Task1 test feature normalization metadata is invalid')

    x_task1 = tensors['x_task1']
    weight_before = tensors['weight_before']
    weight_after = tensors['weight_after']
    test_features = tensors['test_features']
    labels = tensors['test_labels'].long()
    comparison._assert_close(
        tensors['raw_gram'], x_task1.T @ x_task1, 'Raw Gram',
    )
    reconstructed_raw_weight = weight_before.clone()
    reconstructed_raw_weight[:comparison.TASK1_CLASS_COUNT], _ = project_linear_weight(
        weight_before[:comparison.TASK1_CLASS_COUNT],
        tensors['raw_projection'],
    )
    comparison._assert_close(
        reconstructed_raw_weight, weight_after, 'Raw SAP weight',
    )
    pre_logits = torch.nn.functional.linear(test_features, weight_before, bias)
    raw_logits = torch.nn.functional.linear(test_features, weight_after, bias)
    comparison._assert_close(pre_logits, tensors['pre_logits'], 'Pre-SAP logits')
    comparison._assert_close(raw_logits, tensors['raw_logits'], 'Raw SAP logits')
    pre_seen = pre_logits[:, :comparison.TASK1_CLASS_COUNT]
    raw_seen = raw_logits[:, :comparison.TASK1_CLASS_COUNT]
    pre_predictions = pre_seen.argmax(dim=1)
    raw_predictions = raw_seen.argmax(dim=1)
    if not torch.equal(pre_predictions, tensors['pre_predictions'].long()):
        raise AssertionError('Pre-SAP prediction reconstruction mismatch')
    if not torch.equal(raw_predictions, tensors['raw_predictions'].long()):
        raise AssertionError('Raw SAP prediction reconstruction mismatch')
    if abs(
        comparison._accuracy(pre_predictions, labels)
        - float(decision_stats['pre_accuracy_evaluator'])
    ) > sanity_tolerance:
        raise AssertionError('Pre-SAP evaluator accuracy reconstruction mismatch')
    if abs(
        comparison._accuracy(raw_predictions, labels)
        - float(decision_stats['post_accuracy_evaluator'])
    ) > sanity_tolerance:
        raise AssertionError('Raw SAP evaluator accuracy reconstruction mismatch')

    centered = comparison.build_centered_candidate(
        x_task1,
        weight_before,
        scale=comparison.SAP_SCALE,
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
    )
    centered_logits = torch.nn.functional.linear(
        test_features, centered['weight_centered'], bias,
    )
    centered_seen = centered_logits[:, :comparison.TASK1_CLASS_COUNT]
    overall = {}
    for candidate, logits in (
        ('pre_sap', pre_seen),
        ('raw_sap', raw_seen),
        ('centered_sap', centered_seen),
    ):
        margins = comparison.true_class_margin(logits, labels)
        overall[candidate] = comparison._candidate_summary(logits, labels, margins)
    _assert_overall_summary(
        overall,
        existing_summary,
        tolerance=sanity_tolerance,
    )

    result = analyze_class4_decisions(labels, pre_seen, raw_seen, centered_seen)
    result['summary'].update({
        'decomposition_device': centered['decomposition_device'],
        'decomposition_dtype': centered['decomposition_dtype'],
        'sap_scale': comparison.SAP_SCALE,
        'second_l2_normalization': False,
        'overall_comparison_summary': str(existing_summary_path),
        'overall_comparison_sanity': 'pass',
    })
    rows = _detail_rows(result)

    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / 'class4_decision_summary.json').write_text(
        json.dumps(result['summary'], indent=2, sort_keys=True),
        encoding='utf-8',
    )
    with (output_directory / 'class4_sample_details.csv').open(
        'w', newline='', encoding='utf-8',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_DETAIL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _save_figure(
        output_directory / 'class4_pre_raw_centered_decision_analysis.png',
        result,
    )
    return output_directory


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--comparison-summary', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--decomposition-device', choices=('cpu', 'cuda'))
    parser.add_argument(
        '--decomposition-dtype', choices=('artifact', 'float32', 'float64'),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    output = run_class4_analysis(
        arguments.artifact_dir,
        arguments.output_dir,
        comparison_summary_path=arguments.comparison_summary,
        decomposition_device=arguments.decomposition_device,
        decomposition_dtype=arguments.decomposition_dtype,
    )
    print(json.dumps({'status': 'success', 'output_directory': str(output)}))


if __name__ == '__main__':
    main()
