#!/usr/bin/env python3
"""Offline paired Task1 comparison of Raw SAP and feature-centered SAP."""

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

from scripts.analyze_first_session_sap import (  # noqa: E402
    _resolve_decomposition_backend,
)
from utils.sap import (  # noqa: E402
    build_sap_projection_from_gram,
    project_linear_weight,
)


TASK1_CLASS_COUNT = 10
SAP_SCALE = 3000.0
REQUIRED_TENSORS = {
    'x_task1': 'X_task1.pt',
    'raw_gram': 'G_task_0.pt',
    'raw_projection': 'M_task_0.pt',
    'weight_before': 'W_before.pt',
    'weight_after': 'W_after.pt',
    'test_features': 'task1_test_features_raw.pt',
    'test_labels': 'task1_test_labels.pt',
    'pre_logits': 'task1_test_logits_pre_sap.pt',
    'raw_logits': 'task1_test_logits_post_sap.pt',
    'pre_predictions': 'task1_test_predictions_pre_sap.pt',
    'raw_predictions': 'task1_test_predictions_post_sap.pt',
}


def _load_tensor(path: Path) -> Tensor:
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, Tensor):
        raise TypeError(f'{path.name} must contain a tensor')
    return value


def _accuracy(predictions: Tensor, labels: Tensor) -> float:
    return float((predictions == labels).sum().item() / labels.numel() * 100.0)


def build_centered_candidate(
    x_task1: Tensor,
    weight_before: Tensor,
    *,
    class_count: int = TASK1_CLASS_COUNT,
    scale: float = SAP_SCALE,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
    norm_tolerance: float = 1e-5,
) -> dict:
    """Build Centered SAP once from the saved normalized Task1 reference."""
    if float(scale) != SAP_SCALE:
        raise ValueError('Task1 Raw-vs-Centered comparison requires alpha=3000')
    if x_task1.ndim != 2 or not x_task1.is_floating_point():
        raise ValueError('X_task1 must be a floating-point matrix')
    if weight_before.ndim != 2 or weight_before.shape[1] != x_task1.shape[1]:
        raise ValueError('W_before does not match X_task1 feature dimension')
    if not 0 < class_count <= weight_before.shape[0]:
        raise ValueError('invalid seen-class count')
    if not torch.isfinite(x_task1).all() or not torch.isfinite(weight_before).all():
        raise ValueError('centered candidate inputs contain NaN or Inf')
    row_norms = x_task1.norm(p=2, dim=1)
    torch.testing.assert_close(
        row_norms,
        torch.ones_like(row_norms),
        rtol=norm_tolerance,
        atol=norm_tolerance,
    )

    device, dtype = _resolve_decomposition_backend(
        x_task1.T @ x_task1,
        decomposition_device,
        decomposition_dtype,
    )
    features = x_task1.detach().to(device=device, dtype=dtype)
    feature_mean = features.mean(dim=0, keepdim=True)
    centered_features = features - feature_mean
    centered_gram = centered_features.T @ centered_features
    if centered_gram.trace() <= 0 or not torch.isfinite(centered_gram).all():
        raise ValueError('Centered Gram must have finite positive energy')
    centered_projection = build_sap_projection_from_gram(
        centered_gram,
        scale=float(scale),
    )

    weight_device = weight_before.detach().to(device=device)
    centered_seen, _ = project_linear_weight(
        weight_device[:class_count],
        centered_projection,
    )
    centered_weight_device = weight_device.clone()
    centered_weight_device[:class_count] = centered_seen
    centered_weight = centered_weight_device.to(
        device=weight_before.device,
        dtype=weight_before.dtype,
    )
    if not torch.equal(centered_weight[class_count:], weight_before[class_count:]):
        raise AssertionError('Centered SAP changed unseen classifier rows')
    return {
        'feature_mean': feature_mean,
        'centered_features': centered_features,
        'centered_gram': centered_gram,
        'centered_projection': centered_projection,
        'weight_centered': centered_weight,
        'decomposition_device': str(device),
        'decomposition_dtype': str(dtype).removeprefix('torch.'),
        'second_l2_normalization': False,
    }


def true_class_margin(logits: Tensor, labels: Tensor) -> Tensor:
    """Return true-class logit minus the strongest competing-class logit."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError('logits and labels must align by sample')
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError('labels fall outside the evaluated classes')
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    return true_logits - competitors.max(dim=1).values


def paired_candidate_statistics(
    labels: Tensor,
    pre_predictions: Tensor,
    candidate_predictions: Tensor,
    margin_delta: Tensor,
) -> dict[str, float | int]:
    if not (
        labels.shape == pre_predictions.shape
        == candidate_predictions.shape == margin_delta.shape
    ):
        raise ValueError('paired statistics require aligned one-dimensional tensors')
    pre_correct = pre_predictions == labels
    candidate_correct = candidate_predictions == labels
    changed = candidate_predictions != pre_predictions
    return {
        'mean_delta_margin': float(margin_delta.mean().item()),
        'median_delta_margin': float(margin_delta.median().item()),
        'margin_improved_count': int((margin_delta > 0).sum().item()),
        'margin_degraded_count': int((margin_delta < 0).sum().item()),
        'prediction_changed_count': int(changed.sum().item()),
        'correct_to_wrong': int((pre_correct & ~candidate_correct).sum().item()),
        'wrong_to_correct': int((~pre_correct & candidate_correct).sum().item()),
        'wrong_to_wrong_prediction_changed': int(
            (~pre_correct & ~candidate_correct & changed).sum().item()
        ),
    }


def _candidate_summary(logits: Tensor, labels: Tensor, margins: Tensor) -> dict:
    predictions = logits.argmax(dim=1)
    return {
        'accuracy': _accuracy(predictions, labels),
        'mean_margin': float(margins.mean().item()),
        'median_margin': float(margins.median().item()),
    }


def _assert_close(actual: Tensor, expected: Tensor, description: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    except AssertionError as error:
        raise AssertionError(f'{description} reconstruction mismatch') from error


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(
    output_directory: Path,
    delta_raw: Tensor,
    delta_centered: Tensor,
    raw_changed: Tensor,
    centered_changed: Tensor,
    per_class_rows: list[dict],
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _save_plots_with_pillow(
            output_directory,
            delta_raw,
            delta_centered,
            raw_changed,
            centered_changed,
            per_class_rows,
        )
        return

    raw = delta_raw.detach().cpu().numpy()
    centered = delta_centered.detach().cpu().numpy()

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(raw, bins=40, alpha=0.55, label='Raw SAP')
    axis.hist(centered, bins=40, alpha=0.55, label='Centered SAP')
    axis.axvline(0, color='black', linewidth=1)
    axis.set(xlabel='Margin change vs Pre-SAP', ylabel='Sample count')
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory / '01_raw_vs_centered_margin_delta_distribution.png',
        dpi=180,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    flip = (raw_changed | centered_changed).detach().cpu().numpy()
    axis.scatter(raw[~flip], centered[~flip], s=14, alpha=0.55, label='No flip')
    axis.scatter(
        raw[flip], centered[flip], s=30, facecolors='none', edgecolors='red',
        label='Prediction flip',
    )
    lower = min(float(raw.min()), float(centered.min()))
    upper = max(float(raw.max()), float(centered.max()))
    axis.plot([lower, upper], [lower, upper], '--', color='black', linewidth=1)
    axis.set(xlabel='Raw SAP margin change', ylabel='Centered SAP margin change')
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory / '02_raw_vs_centered_sample_margin_delta.png', dpi=180,
    )
    plt.close(figure)

    class_ids = [row['class_id'] for row in per_class_rows]
    raw_class = [row['raw_delta_margin'] for row in per_class_rows]
    centered_class = [row['centered_delta_margin'] for row in per_class_rows]
    positions = torch.arange(len(class_ids), dtype=torch.float32).numpy()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - 0.2, raw_class, width=0.4, label='Raw SAP')
    axis.bar(positions + 0.2, centered_class, width=0.4, label='Centered SAP')
    axis.axhline(0, color='black', linewidth=1)
    axis.set(
        xlabel='Task1 class', ylabel='Mean margin change vs Pre-SAP',
        xticks=positions, xticklabels=class_ids,
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory / '03_raw_vs_centered_per_class_margin_delta.png', dpi=180,
    )
    plt.close(figure)


def _save_plots_with_pillow(
    output_directory: Path,
    delta_raw: Tensor,
    delta_centered: Tensor,
    raw_changed: Tensor,
    centered_changed: Tensor,
    per_class_rows: list[dict],
) -> None:
    """Dependency-light PNG fallback for environments without matplotlib."""
    from PIL import Image, ImageDraw

    width, height = 900, 600
    left, top, right, bottom = 90, 45, 35, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    raw = delta_raw.detach().cpu().float()
    centered = delta_centered.detach().cpu().float()

    def canvas(title: str, x_label: str, y_label: str):
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((left, 15), title, fill='black')
        draw.line((left, top, left, top + plot_height), fill='black', width=2)
        draw.line(
            (left, top + plot_height, left + plot_width, top + plot_height),
            fill='black', width=2,
        )
        draw.text((left + plot_width // 3, height - 35), x_label, fill='black')
        draw.text((8, top + plot_height // 2), y_label, fill='black')
        return image, draw

    def map_x(value: float, minimum: float, maximum: float) -> float:
        span = maximum - minimum or 1.0
        return left + (value - minimum) / span * plot_width

    def map_y(value: float, minimum: float, maximum: float) -> float:
        span = maximum - minimum or 1.0
        return top + plot_height - (value - minimum) / span * plot_height

    all_delta = torch.cat((raw, centered))
    minimum, maximum = float(all_delta.min()), float(all_delta.max())
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    bins = 40
    raw_hist = torch.histc(raw, bins=bins, min=minimum, max=maximum)
    centered_hist = torch.histc(centered, bins=bins, min=minimum, max=maximum)
    count_max = max(float(raw_hist.max()), float(centered_hist.max()), 1.0)
    image, draw = canvas(
        'Raw vs Feature-centered SAP margin delta distribution',
        'Margin change vs Pre-SAP', 'Sample count',
    )
    bin_width = plot_width / bins
    for index in range(bins):
        x0 = left + index * bin_width
        raw_height = float(raw_hist[index]) / count_max * plot_height
        centered_height = float(centered_hist[index]) / count_max * plot_height
        draw.rectangle(
            (x0, top + plot_height - raw_height, x0 + bin_width * 0.48, top + plot_height),
            fill='#0072B2',
        )
        draw.rectangle(
            (x0 + bin_width * 0.5, top + plot_height - centered_height,
             x0 + bin_width * 0.98, top + plot_height),
            fill='#D55E00',
        )
    zero_x = map_x(0.0, minimum, maximum)
    draw.line((zero_x, top, zero_x, top + plot_height), fill='#444444', width=1)
    draw.text((width - 245, 18), 'Raw SAP', fill='#0072B2')
    draw.text((width - 150, 18), 'Centered SAP', fill='#D55E00')
    image.save(output_directory / '01_raw_vs_centered_margin_delta_distribution.png')

    image, draw = canvas(
        'Per-sample Raw vs Feature-centered margin change',
        'Raw SAP margin change', 'Centered SAP margin change',
    )
    draw.line(
        (map_x(minimum, minimum, maximum), map_y(minimum, minimum, maximum),
         map_x(maximum, minimum, maximum), map_y(maximum, minimum, maximum)),
        fill='#555555', width=2,
    )
    flip = (raw_changed | centered_changed).detach().cpu()
    for x_value, y_value, is_flip in zip(raw.tolist(), centered.tolist(), flip.tolist()):
        x_pixel = map_x(x_value, minimum, maximum)
        y_pixel = map_y(y_value, minimum, maximum)
        color = '#D55E00' if is_flip else '#0072B2'
        radius = 5 if is_flip else 3
        draw.ellipse(
            (x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius),
            outline=color, fill=None if is_flip else color, width=2,
        )
    image.save(output_directory / '02_raw_vs_centered_sample_margin_delta.png')

    raw_class = [float(row['raw_delta_margin']) for row in per_class_rows]
    centered_class = [float(row['centered_delta_margin']) for row in per_class_rows]
    class_values = raw_class + centered_class + [0.0]
    class_min, class_max = min(class_values), max(class_values)
    if class_min == class_max:
        class_min -= 1.0
        class_max += 1.0
    image, draw = canvas(
        'Per-class mean margin change', 'Task1 class', 'Mean margin change',
    )
    zero_y = map_y(0.0, class_min, class_max)
    draw.line((left, zero_y, left + plot_width, zero_y), fill='#444444', width=1)
    group_width = plot_width / len(per_class_rows)
    for index, row in enumerate(per_class_rows):
        center = left + (index + 0.5) * group_width
        for offset, value, color in (
            (-0.22, raw_class[index], '#0072B2'),
            (0.02, centered_class[index], '#D55E00'),
        ):
            x0 = center + offset * group_width
            x1 = x0 + group_width * 0.2
            value_y = map_y(value, class_min, class_max)
            draw.rectangle((x0, min(zero_y, value_y), x1, max(zero_y, value_y)), fill=color)
        draw.text((center - 4, top + plot_height + 8), str(row['class_id']), fill='black')
    image.save(output_directory / '03_raw_vs_centered_per_class_margin_delta.png')

def run_comparison(
    artifact_directory: Path,
    output_directory: Path | None = None,
    *,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
    sanity_tolerance: float = 1e-5,
) -> Path:
    """Run the frozen-state paired comparison from one Task1 artifact set."""
    artifact_directory = artifact_directory.expanduser().resolve()
    if output_directory is None:
        output_directory = artifact_directory / 'task1_raw_vs_centered_sap'
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(f'output directory already exists: {output_directory}')

    paths = {
        name: artifact_directory / filename
        for name, filename in REQUIRED_TENSORS.items()
    }
    paths['manifest'] = artifact_directory / 'manifest.json'
    paths['decision_stats'] = artifact_directory / 'task1_test_decision_stats.json'
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing Task1 comparison inputs: ' + ', '.join(missing))
    tensors = {name: _load_tensor(path) for name, path in paths.items() if name not in {'manifest', 'decision_stats'}}
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    decision_stats = json.loads(paths['decision_stats'].read_text(encoding='utf-8'))

    classifier_has_bias = bool(manifest.get('classifier_has_bias', False))
    bias_path = artifact_directory / 'classifier_bias_before.pt'
    if classifier_has_bias:
        if not bias_path.is_file():
            raise FileNotFoundError(f'missing Task1 classifier bias: {bias_path}')
        bias = _load_tensor(bias_path)
    else:
        bias = None
    if manifest.get('test_feature_type') != 'raw_classifier_input':
        raise ValueError('Task1 test feature artifact is not raw classifier input')
    if manifest.get('test_feature_l2_normalized') is not False:
        raise ValueError('Task1 test feature artifact normalization metadata is invalid')

    x_task1 = tensors['x_task1']
    raw_gram = tensors['raw_gram']
    raw_projection = tensors['raw_projection']
    weight_before = tensors['weight_before']
    weight_after = tensors['weight_after']
    test_features = tensors['test_features']
    labels = tensors['test_labels'].long()
    if weight_before.shape[0] < TASK1_CLASS_COUNT:
        raise ValueError('classifier does not contain all Task1 seen classes')
    if test_features.ndim != 2 or test_features.shape[1] != weight_before.shape[1]:
        raise ValueError('raw test feature shape does not match classifier')
    sample_count = labels.numel()
    aligned = [
        tensors[name].shape[0]
        for name in ('test_features', 'pre_logits', 'raw_logits', 'pre_predictions', 'raw_predictions')
    ]
    if labels.ndim != 1 or any(count != sample_count for count in aligned):
        raise ValueError('Task1 test artifacts are not sample-aligned')
    if labels.min() < 0 or labels.max() >= TASK1_CLASS_COUNT:
        raise ValueError('Task1 test labels fall outside seen classes 0-9')

    expected_raw_gram = x_task1.T @ x_task1
    _assert_close(raw_gram, expected_raw_gram, 'Raw Gram')
    reconstructed_raw_weight = weight_before.clone()
    reconstructed_raw_weight[:TASK1_CLASS_COUNT], _ = project_linear_weight(
        weight_before[:TASK1_CLASS_COUNT], raw_projection,
    )
    _assert_close(reconstructed_raw_weight, weight_after, 'Raw SAP weight')
    if not torch.equal(weight_after[TASK1_CLASS_COUNT:], weight_before[TASK1_CLASS_COUNT:]):
        raise AssertionError('Raw SAP changed unseen classifier rows')

    pre_logits = torch.nn.functional.linear(test_features, weight_before, bias)
    raw_logits = torch.nn.functional.linear(test_features, weight_after, bias)
    _assert_close(pre_logits, tensors['pre_logits'], 'Pre-SAP logits')
    _assert_close(raw_logits, tensors['raw_logits'], 'Raw SAP logits')
    pre_seen = pre_logits[:, :TASK1_CLASS_COUNT]
    raw_seen = raw_logits[:, :TASK1_CLASS_COUNT]
    pre_predictions = pre_seen.argmax(dim=1)
    raw_predictions = raw_seen.argmax(dim=1)
    if not torch.equal(pre_predictions, tensors['pre_predictions'].long()):
        raise AssertionError('recomputed Pre-SAP predictions differ from saved predictions')
    if not torch.equal(raw_predictions, tensors['raw_predictions'].long()):
        raise AssertionError('recomputed Raw SAP predictions differ from saved predictions')
    pre_accuracy = _accuracy(pre_predictions, labels)
    raw_accuracy = _accuracy(raw_predictions, labels)
    if abs(pre_accuracy - float(decision_stats['pre_accuracy_evaluator'])) > sanity_tolerance:
        raise AssertionError('recomputed Pre-SAP accuracy differs from evaluator accuracy')
    if abs(raw_accuracy - float(decision_stats['post_accuracy_evaluator'])) > sanity_tolerance:
        raise AssertionError('recomputed Raw SAP accuracy differs from evaluator accuracy')

    centered = build_centered_candidate(
        x_task1,
        weight_before,
        scale=SAP_SCALE,
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
    )
    weight_centered = centered['weight_centered']
    centered_logits = torch.nn.functional.linear(test_features, weight_centered, bias)
    centered_seen = centered_logits[:, :TASK1_CLASS_COUNT]
    centered_predictions = centered_seen.argmax(dim=1)

    margin_pre = true_class_margin(pre_seen, labels)
    margin_raw = true_class_margin(raw_seen, labels)
    margin_centered = true_class_margin(centered_seen, labels)
    delta_raw = margin_raw - margin_pre
    delta_centered = margin_centered - margin_pre
    raw_changed = raw_predictions != pre_predictions
    centered_changed = centered_predictions != pre_predictions

    summary = {
        'sample_count': int(sample_count),
        'seen_classes': list(range(TASK1_CLASS_COUNT)),
        'sap_scale': SAP_SCALE,
        'decomposition_device': centered['decomposition_device'],
        'decomposition_dtype': centered['decomposition_dtype'],
        'second_l2_normalization': False,
        'pre_sap': _candidate_summary(pre_seen, labels, margin_pre),
        'raw_sap': _candidate_summary(raw_seen, labels, margin_raw),
        'centered_sap': _candidate_summary(centered_seen, labels, margin_centered),
        'raw_vs_pre': paired_candidate_statistics(
            labels, pre_predictions, raw_predictions, delta_raw,
        ),
        'centered_vs_pre': paired_candidate_statistics(
            labels, pre_predictions, centered_predictions, delta_centered,
        ),
        'sanity_checks': {
            'reference_rows_l2_normalized': 'pass',
            'raw_gram_reconstructed': 'pass',
            'raw_weight_reconstructed': 'pass',
            'pre_logits_reconstructed': 'pass',
            'raw_logits_reconstructed': 'pass',
            'pre_predictions_reconstructed': 'pass',
            'raw_predictions_reconstructed': 'pass',
            'pre_accuracy_reconstructed': 'pass',
            'raw_accuracy_reconstructed': 'pass',
            'unseen_rows_bitwise_unchanged': 'pass',
            'bias_unchanged': 'pass',
        },
    }

    per_class_rows = []
    for class_id in range(TASK1_CLASS_COUNT):
        mask = labels == class_id
        per_class_rows.append({
            'class_id': class_id,
            'sample_count': int(mask.sum().item()),
            'pre_mean_margin': float(margin_pre[mask].mean().item()),
            'raw_mean_margin': float(margin_raw[mask].mean().item()),
            'centered_mean_margin': float(margin_centered[mask].mean().item()),
            'raw_delta_margin': float(delta_raw[mask].mean().item()),
            'centered_delta_margin': float(delta_centered[mask].mean().item()),
            'raw_prediction_changed_count': int(raw_changed[mask].sum().item()),
            'centered_prediction_changed_count': int(centered_changed[mask].sum().item()),
        })
    changed_rows = []
    changed_mask = raw_changed | centered_changed
    for index in torch.where(changed_mask)[0].tolist():
        changed_rows.append({
            'sample_index': index,
            'true_label': int(labels[index].item()),
            'pre_prediction': int(pre_predictions[index].item()),
            'raw_prediction': int(raw_predictions[index].item()),
            'centered_prediction': int(centered_predictions[index].item()),
            'pre_margin': float(margin_pre[index].item()),
            'raw_margin': float(margin_raw[index].item()),
            'centered_margin': float(margin_centered[index].item()),
            'delta_raw': float(delta_raw[index].item()),
            'delta_centered': float(delta_centered[index].item()),
            'raw_prediction_changed': bool(raw_changed[index].item()),
            'centered_prediction_changed': bool(centered_changed[index].item()),
        })

    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / 'raw_vs_centered_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8',
    )
    _write_csv(
        output_directory / 'raw_vs_centered_per_class.csv',
        list(per_class_rows[0]),
        per_class_rows,
    )
    changed_fields = [
        'sample_index', 'true_label', 'pre_prediction', 'raw_prediction',
        'centered_prediction', 'pre_margin', 'raw_margin', 'centered_margin',
        'delta_raw', 'delta_centered', 'raw_prediction_changed',
        'centered_prediction_changed',
    ]
    _write_csv(
        output_directory / 'raw_vs_centered_changed_samples.csv',
        changed_fields,
        changed_rows,
    )
    _save_plots(
        output_directory,
        delta_raw,
        delta_centered,
        raw_changed,
        centered_changed,
        per_class_rows,
    )
    return output_directory


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--decomposition-device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument(
        '--decomposition-dtype', default='artifact',
        choices=('artifact', 'float32', 'float64'),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    output = run_comparison(
        arguments.artifact_dir,
        arguments.output_dir,
        decomposition_device=arguments.decomposition_device,
        decomposition_dtype=arguments.decomposition_dtype,
    )
    print(json.dumps({'status': 'success', 'output_directory': str(output)}))


if __name__ == '__main__':
    main()
