#!/usr/bin/env python3
"""Joint t-SNE of Task1 Pre/Raw/Centered SAP effective features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.manifold import TSNE
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.compare_task1_raw_vs_centered_sap as comparison  # noqa: E402
from utils.sap import project_linear_weight  # noqa: E402


RANDOM_STATE = 0
PERPLEXITY = 30.0
TSNE_INIT = 'pca'
TSNE_LEARNING_RATE = 'auto'
COLORS = (
    '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
    '#D55E00', '#CC79A7', '#000000', '#999999', '#7B61A8',
)


def build_effective_features(features: Tensor, projection: Tensor) -> Tensor:
    """Apply the feature-side operator equivalent to ``W @ M.T``."""
    if features.ndim != 2 or projection.ndim != 2:
        raise ValueError('features and projection must both be matrices')
    feature_dimension = features.shape[1]
    if projection.shape != (feature_dimension, feature_dimension):
        raise ValueError('projection shape does not match feature dimension')
    if not features.is_floating_point() or not projection.is_floating_point():
        raise TypeError('effective-feature inputs must use floating-point dtypes')
    if not torch.isfinite(features).all() or not torch.isfinite(projection).all():
        raise ValueError('effective-feature inputs contain NaN or Inf')
    operator = projection.to(device=features.device, dtype=features.dtype)
    return features @ operator


def _relative_error(actual: Tensor, expected: Tensor) -> float:
    if actual.shape != expected.shape:
        raise ValueError('relative-error tensors have different shapes')
    denominator = expected.norm().clamp_min(torch.finfo(expected.dtype).eps)
    return float(((actual - expected).norm() / denominator).item())


def _assert_logits(
    actual: Tensor,
    expected: Tensor,
    description: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> float:
    error = _relative_error(actual, expected)
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exception:
        raise AssertionError(
            f'{description} reconstruction mismatch: relative_error={error:.6e}'
        ) from exception
    return error


def fit_joint_tsne(
    pre_features: Tensor,
    raw_features: Tensor,
    centered_features: Tensor,
    *,
    random_state: int = RANDOM_STATE,
    perplexity: float = PERPLEXITY,
) -> dict[str, Tensor]:
    """Fit exactly one t-SNE over aligned Pre/Raw/Centered sample blocks."""
    if not (
        pre_features.shape == raw_features.shape == centered_features.shape
        and pre_features.ndim == 2
        and pre_features.shape[0] > 0
    ):
        raise ValueError('joint t-SNE requires three aligned feature matrices')
    if not all(
        torch.isfinite(value).all()
        for value in (pre_features, raw_features, centered_features)
    ):
        raise ValueError('joint t-SNE inputs contain NaN or Inf')
    joint = torch.cat((pre_features, raw_features, centered_features), dim=0)
    if perplexity >= joint.shape[0]:
        raise ValueError('t-SNE perplexity must be smaller than joint sample count')
    estimator = TSNE(
        n_components=2,
        random_state=int(random_state),
        perplexity=float(perplexity),
        init=TSNE_INIT,
        learning_rate=TSNE_LEARNING_RATE,
    )
    embedded = estimator.fit_transform(joint.detach().cpu().numpy())
    embedding = torch.as_tensor(embedded)
    sample_count = pre_features.shape[0]
    return {
        'pre': embedding[:sample_count].clone(),
        'raw': embedding[sample_count:2 * sample_count].clone(),
        'centered': embedding[2 * sample_count:].clone(),
    }


def _save_plot(output_path: Path, embedding: dict[str, Tensor], labels: Tensor) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        _save_plot_with_pillow(output_path, embedding, labels)
        return

    all_coordinates = torch.cat(tuple(embedding.values()), dim=0)
    x_min, x_max = map(float, (all_coordinates[:, 0].min(), all_coordinates[:, 0].max()))
    y_min, y_max = map(float, (all_coordinates[:, 1].min(), all_coordinates[:, 1].max()))
    x_padding = max((x_max - x_min) * 0.04, 1e-6)
    y_padding = max((y_max - y_min) * 0.04, 1e-6)
    titles = (
        ('pre', 'A. Pre-SAP'),
        ('raw', 'B. Raw SAP, alpha=3000'),
        ('centered', 'C. Centered SAP, alpha=3000'),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    labels_cpu = labels.detach().cpu()
    for axis, (key, title) in zip(axes, titles):
        coordinates = embedding[key].detach().cpu()
        for class_id in range(comparison.TASK1_CLASS_COUNT):
            mask = labels_cpu == class_id
            axis.scatter(
                coordinates[mask, 0], coordinates[mask, 1],
                s=11, alpha=0.72, color=COLORS[class_id], linewidths=0,
            )
        axis.set_title(title)
        axis.set_xlim(x_min - x_padding, x_max + x_padding)
        axis.set_ylim(y_min - y_padding, y_max + y_padding)
        axis.set_xlabel('Joint t-SNE dimension 1')
    axes[0].set_ylabel('Joint t-SNE dimension 2')
    handles = [
        Line2D(
            [0], [0], marker='o', linestyle='', color=COLORS[class_id],
            label=f'Class {class_id}', markersize=5,
        )
        for class_id in range(comparison.TASK1_CLASS_COUNT)
    ]
    figure.legend(
        handles=handles, loc='lower center', ncol=10, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(figure)


def _save_plot_with_pillow(
    output_path: Path,
    embedding: dict[str, Tensor],
    labels: Tensor,
) -> None:
    """Dependency-light three-panel PNG fallback for test environments."""
    from PIL import Image, ImageDraw

    width, height = 1500, 520
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    all_coordinates = torch.cat(tuple(embedding.values()), dim=0).float()
    x_min, x_max = float(all_coordinates[:, 0].min()), float(all_coordinates[:, 0].max())
    y_min, y_max = float(all_coordinates[:, 1].min()), float(all_coordinates[:, 1].max())
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    panel_width, panel_height, top = 420, 370, 50
    panel_lefts = (55, 540, 1025)
    titles = (
        ('pre', 'A. Pre-SAP'),
        ('raw', 'B. Raw SAP, alpha=3000'),
        ('centered', 'C. Centered SAP, alpha=3000'),
    )
    labels_cpu = labels.detach().cpu().tolist()

    def map_point(point: Tensor, left: int) -> tuple[float, float]:
        x = left + (float(point[0]) - x_min) / (x_max - x_min) * panel_width
        y = top + panel_height - (
            (float(point[1]) - y_min) / (y_max - y_min) * panel_height
        )
        return x, y

    for left, (key, title) in zip(panel_lefts, titles):
        draw.text((left, 18), title, fill='black')
        draw.rectangle(
            (left, top, left + panel_width, top + panel_height),
            outline='black', width=2,
        )
        for point, class_id in zip(embedding[key], labels_cpu):
            x, y = map_point(point, left)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=COLORS[class_id])
    for class_id in range(comparison.TASK1_CLASS_COUNT):
        x = 270 + class_id * 100
        draw.ellipse((x, 465, x + 10, 475), fill=COLORS[class_id])
        draw.text((x + 14, 463), str(class_id), fill='black')
    image.save(output_path)


def run_visualization(
    artifact_directory: Path,
    output_directory: Path | None = None,
    *,
    decomposition_device: str = 'cpu',
    decomposition_dtype: str = 'artifact',
    sanity_tolerance: float = 1e-5,
) -> Path:
    """Reconstruct effective features, validate logits, and run joint t-SNE."""
    artifact_directory = artifact_directory.expanduser().resolve()
    if output_directory is None:
        output_directory = artifact_directory / 'task1_sap_joint_tsne'
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
        raise FileNotFoundError('missing Task1 t-SNE inputs: ' + ', '.join(missing))
    tensors = {
        name: comparison._load_tensor(path)
        for name, path in paths.items()
        if name not in {'manifest', 'decision_stats'}
    }
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    decision_stats = json.loads(paths['decision_stats'].read_text(encoding='utf-8'))
    bias = None
    bias_path = artifact_directory / 'classifier_bias_before.pt'
    if bool(manifest.get('classifier_has_bias', False)):
        if not bias_path.is_file():
            raise FileNotFoundError(f'missing classifier bias: {bias_path}')
        bias = comparison._load_tensor(bias_path)
    if manifest.get('test_feature_type') != 'raw_classifier_input':
        raise ValueError('Task1 t-SNE requires raw classifier-input test features')
    if manifest.get('test_feature_l2_normalized') is not False:
        raise ValueError('Task1 test feature normalization metadata is invalid')

    x_reference = tensors['x_task1']
    x_pre = tensors['test_features']
    labels = tensors['test_labels'].long()
    weight_before = tensors['weight_before']
    weight_after = tensors['weight_after']
    sample_count = x_pre.shape[0]
    if (
        x_pre.ndim != 2
        or labels.ndim != 1
        or labels.shape[0] != sample_count
        or x_pre.shape[1] != weight_before.shape[1]
    ):
        raise ValueError('Task1 test features, labels, and classifier do not align')
    classes = labels.unique(sorted=True).tolist()
    expected_classes = list(range(comparison.TASK1_CLASS_COUNT))
    if classes != expected_classes:
        raise ValueError('Task1 t-SNE requires exactly seen classes 0-9')
    for name in ('pre_logits', 'raw_logits', 'pre_predictions', 'raw_predictions'):
        if tensors[name].shape[0] != sample_count:
            raise ValueError(f'{name} is not aligned with Task1 test samples')

    comparison._assert_close(
        tensors['raw_gram'], x_reference.T @ x_reference, 'Raw Gram',
    )
    reconstructed_raw_weight = weight_before.clone()
    reconstructed_raw_weight[:comparison.TASK1_CLASS_COUNT], _ = project_linear_weight(
        weight_before[:comparison.TASK1_CLASS_COUNT],
        tensors['raw_projection'],
    )
    comparison._assert_close(
        reconstructed_raw_weight, weight_after, 'Raw SAP weight',
    )
    pre_logits = torch.nn.functional.linear(x_pre, weight_before, bias)
    raw_weight_logits = torch.nn.functional.linear(x_pre, weight_after, bias)
    comparison._assert_close(pre_logits, tensors['pre_logits'], 'Pre-SAP logits')
    comparison._assert_close(
        raw_weight_logits, tensors['raw_logits'], 'Raw SAP logits',
    )
    pre_seen = pre_logits[:, :comparison.TASK1_CLASS_COUNT]
    raw_seen = raw_weight_logits[:, :comparison.TASK1_CLASS_COUNT]
    if not torch.equal(
        pre_seen.argmax(dim=1), tensors['pre_predictions'].long(),
    ):
        raise AssertionError('Pre-SAP saved predictions do not use classes 0-9')
    if not torch.equal(
        raw_seen.argmax(dim=1), tensors['raw_predictions'].long(),
    ):
        raise AssertionError('Raw SAP saved predictions do not use classes 0-9')
    if abs(
        comparison._accuracy(pre_seen.argmax(dim=1), labels)
        - float(decision_stats['pre_accuracy_evaluator'])
    ) > sanity_tolerance:
        raise AssertionError('Pre-SAP accuracy reconstruction mismatch')
    if abs(
        comparison._accuracy(raw_seen.argmax(dim=1), labels)
        - float(decision_stats['post_accuracy_evaluator'])
    ) > sanity_tolerance:
        raise AssertionError('Raw SAP accuracy reconstruction mismatch')

    centered = comparison.build_centered_candidate(
        x_reference,
        weight_before,
        scale=comparison.SAP_SCALE,
        decomposition_device=decomposition_device,
        decomposition_dtype=decomposition_dtype,
    )
    centered_projection = centered['centered_projection'].to(
        device=x_pre.device,
        dtype=x_pre.dtype,
    )
    raw_projection = tensors['raw_projection'].to(
        device=x_pre.device,
        dtype=x_pre.dtype,
    )
    x_raw_effective = build_effective_features(x_pre, raw_projection)
    x_centered_effective = build_effective_features(x_pre, centered_projection)
    bias_seen = None if bias is None else bias[:comparison.TASK1_CLASS_COUNT]
    weight_before_seen = weight_before[:comparison.TASK1_CLASS_COUNT]
    raw_effective_logits = torch.nn.functional.linear(
        x_raw_effective, weight_before_seen, bias_seen,
    )
    centered_effective_logits = torch.nn.functional.linear(
        x_centered_effective, weight_before_seen, bias_seen,
    )
    centered_weight_logits = torch.nn.functional.linear(
        x_pre,
        centered['weight_centered'][:comparison.TASK1_CLASS_COUNT],
        bias_seen,
    )
    raw_error = _assert_logits(
        raw_effective_logits,
        raw_seen,
        'Raw SAP logits',
    )
    centered_error = _assert_logits(
        centered_effective_logits,
        centered_weight_logits,
        'Centered SAP logits',
    )

    embedding = fit_joint_tsne(
        x_pre,
        x_raw_effective,
        x_centered_effective,
        random_state=RANDOM_STATE,
        perplexity=PERPLEXITY,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    _save_plot(
        output_directory / '01_task1_pre_raw_centered_joint_tsne.png',
        embedding,
        labels,
    )
    source_paths = {
        name: str(path.resolve())
        for name, path in paths.items()
    }
    if bias is not None:
        source_paths['classifier_bias'] = str(bias_path.resolve())
    summary = {
        'sample_count': int(sample_count),
        'class_count': comparison.TASK1_CLASS_COUNT,
        'classes': expected_classes,
        'alpha': comparison.SAP_SCALE,
        'random_state': RANDOM_STATE,
        'perplexity': PERPLEXITY,
        'init': TSNE_INIT,
        'learning_rate': TSNE_LEARNING_RATE,
        'input_dimension': int(x_pre.shape[1]),
        'joint_embedding_sample_count': int(sample_count * 3),
        'raw_logits_reconstruction_error': raw_error,
        'centered_logits_reconstruction_error': centered_error,
        'second_l2_normalization': False,
        'decomposition_device': centered['decomposition_device'],
        'decomposition_dtype': centered['decomposition_dtype'],
        'source_artifact_paths': source_paths,
        'training_performed': False,
        'sample_order_changed': False,
        'joint_tsne_fit_count': 1,
    }
    (output_directory / 'task1_tsne_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8',
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
    output = run_visualization(
        arguments.artifact_dir,
        arguments.output_dir,
        decomposition_device=arguments.decomposition_device,
        decomposition_dtype=arguments.decomposition_dtype,
    )
    print(json.dumps({'status': 'success', 'output_directory': str(output)}))


if __name__ == '__main__':
    main()
