"""Runtime gates and data adapters for task-boundary SAP."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from utils.sap import (
    RESNET18_LATE_STAGE_CONVS,
    SAPLayerProjectionStats,
    project_resnet18_from_reference_batches,
)
from utils.sap_reference import (
    SAPClassSelectionReport,
    SAPReferenceMemory,
    TRUSTED_SELECTION_SOURCES,
)


SAP_EXECUTED = 'SAP_EXECUTED'
SAP_SKIPPED_INSUFFICIENT_REFERENCE = 'SAP_SKIPPED_INSUFFICIENT_REFERENCE'
SAP_SKIPPED_EXCESSIVE_FALLBACK = 'SAP_SKIPPED_EXCESSIVE_FALLBACK'


@dataclass(frozen=True)
class SAPGateDecision:
    should_execute: bool
    status: str
    current_class_coverage: int
    current_classes_with_five_main: int
    seen_class_coverage: int
    reference_count: int
    fallback_fraction: float


@dataclass(frozen=True)
class SAPProjectionTransaction:
    committed: bool
    accepted: bool
    rejection_reasons: tuple[str, ...]
    minimum_weight_norm_ratio: float
    training_side_accuracy_deltas: dict[str, float]
    layer_stats: dict[str, SAPLayerProjectionStats]
    comparisons: dict[str, 'SAPModelComparison']
    task_accuracy_comparisons: list['SAPTaskAccuracyComparison']
    max_non_target_state_delta: float


@dataclass(frozen=True)
class SAPLossGroupComparison:
    sample_count: int
    mean_loss_before: float
    mean_loss_after: float
    mean_loss_delta: float
    accuracy_before: float
    accuracy_after: float
    accuracy_delta: float
    prediction_flip_rate: float


@dataclass(frozen=True)
class SAPModelComparison:
    sample_count: int
    mean_abs_logits_delta: float
    max_abs_logits_delta: float
    prediction_flip_rate: float
    loss_tertiles: dict[str, SAPLossGroupComparison]


@dataclass(frozen=True)
class SAPTaskAccuracyComparison:
    task_id: int
    sample_count: int
    accuracy_before: float
    accuracy_after: float
    accuracy_delta: float


@dataclass(frozen=True)
class SAPCandidateDecision:
    accepted: bool
    rejection_reasons: tuple[str, ...]
    minimum_weight_norm_ratio: float
    accuracy_deltas: dict[str, float]


def _comparison_accuracy(comparison: SAPModelComparison, *, after: bool) -> float:
    non_empty_groups = [
        group
        for group in comparison.loss_tertiles.values()
        if group.sample_count > 0
    ]
    total = sum(group.sample_count for group in non_empty_groups)
    if total <= 0:
        raise ValueError('SAP comparison contains no samples')
    field = 'accuracy_after' if after else 'accuracy_before'
    return sum(
        group.sample_count * getattr(group, field)
        for group in non_empty_groups
    ) / total


def assess_sap_candidate(
    layer_stats,
    comparisons: dict[str, SAPModelComparison],
    *,
    minimum_weight_norm_ratio: float = 0.25,
    maximum_reference_accuracy_drop: float = 0.01,
    maximum_replay_task_accuracy_drop: float = 0.02,
    minimum_diagnostic_samples: int = 3,
    required_replay_task_names: set[str] | None = None,
) -> SAPCandidateDecision:
    """Gate a candidate using training-side trusted/replay evidence only."""
    minimum_ratio = min(
        (stats.weight_norm_ratio for stats in layer_stats.values()),
        default=1.0,
    )
    reasons = []
    if not isfinite(minimum_ratio):
        reasons.append('NON_FINITE_WEIGHT_NORM_RATIO')
    elif minimum_ratio < minimum_weight_norm_ratio:
        reasons.append('WEIGHT_NORM_RATIO_BELOW_FLOOR')
    accuracy_deltas = {}
    for name, comparison in comparisons.items():
        if comparison.sample_count < minimum_diagnostic_samples:
            reasons.append(f'INSUFFICIENT_DIAGNOSTIC_SAMPLES:{name}')
        delta = _comparison_accuracy(comparison, after=True) - _comparison_accuracy(
            comparison, after=False,
        )
        accuracy_deltas[name] = delta
        if not isfinite(delta):
            reasons.append(f'NON_FINITE_ACCURACY_DELTA:{name}')
        elif name == 'reference' and delta < -maximum_reference_accuracy_drop:
            reasons.append('REFERENCE_ACCURACY_DROP')

    required_replay_task_names = required_replay_task_names or set()
    for name in sorted(required_replay_task_names - comparisons.keys()):
        reasons.append(f'MISSING_REPLAY_TASK:{name}')

    replay_task_names = sorted(name for name in comparisons if name.startswith('replay_task_'))
    replay_names = replay_task_names or (['replay_buffer'] if 'replay_buffer' in comparisons else [])
    for name in replay_names:
        if isfinite(accuracy_deltas[name]) and accuracy_deltas[name] < -maximum_replay_task_accuracy_drop:
            reasons.append(f'REPLAY_TASK_ACCURACY_DROP:{name}')
    return SAPCandidateDecision(
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        minimum_weight_norm_ratio=float(minimum_ratio),
        accuracy_deltas=accuracy_deltas,
    )


def diagnose_reference_purity(references, true_labels_by_sample_id) -> float | None:
    """Return synthetic-noise purity for logging without influencing selection."""
    if true_labels_by_sample_id is None or not references:
        return None
    true_labels = torch.as_tensor(true_labels_by_sample_id, dtype=torch.long).reshape(-1)
    sample_ids = torch.tensor([reference.sample_id for reference in references], dtype=torch.long)
    if sample_ids.min().item() < 0 or sample_ids.max().item() >= len(true_labels):
        return None
    observed = torch.tensor([reference.observed_label for reference in references], dtype=torch.long)
    return (true_labels[sample_ids] == observed).float().mean().item()


def extract_cifar_task_tensors(dataset) -> tuple[Tensor, Tensor, Tensor]:
    """Extract uint8 NCHW images, observed labels, and stable ids from a task dataset."""
    required = ('data', 'targets', 'indexes')
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise ValueError(f'CIFAR SAP task dataset is missing fields: {missing}')
    images = torch.as_tensor(np.asarray(dataset.data))
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f'expected CIFAR NHWC images, got shape {tuple(images.shape)}')
    if images.dtype != torch.uint8:
        raise TypeError('expected raw CIFAR uint8 images')
    labels = torch.as_tensor(dataset.targets, dtype=torch.long).reshape(-1)
    sample_ids = torch.as_tensor(np.asarray(dataset.indexes), dtype=torch.long).reshape(-1)
    if images.shape[0] != labels.numel() or labels.numel() != sample_ids.numel():
        raise ValueError('CIFAR SAP task data fields are not aligned')
    return images.permute(0, 3, 1, 2).contiguous(), labels.clone(), sample_ids.clone()


def validate_reference_gate(
    memory: SAPReferenceMemory,
    current_reports: Sequence[SAPClassSelectionReport],
    *,
    current_class_count: int,
    seen_class_count: int,
) -> SAPGateDecision:
    """Apply the agreed CIFAR100 gate, scaled by the current task class count."""
    if current_class_count <= 0 or seen_class_count <= 0:
        raise ValueError('class counts must be positive')
    current_class_coverage = sum(report.selected_main_count > 0 for report in current_reports)
    current_classes_with_five_main = sum(report.selected_main_count >= 5 for report in current_reports)
    references = memory.items()
    seen_class_coverage = len({reference.observed_label for reference in references})
    fallback_count = sum(
        reference.selection_source not in TRUSTED_SELECTION_SOURCES
        for reference in references
    )
    reference_count = len(references)
    fallback_fraction = fallback_count / reference_count if reference_count else 0.0

    required_current_coverage = ceil(0.9 * current_class_count)
    required_main_classes = ceil(0.8 * current_class_count)
    required_seen_coverage = ceil(0.9 * seen_class_count)
    sufficient = (
        current_class_coverage >= required_current_coverage
        and current_classes_with_five_main >= required_main_classes
        and seen_class_coverage >= required_seen_coverage
        and reference_count >= 4 * seen_class_count
    )
    if fallback_fraction > 0.2:
        status = SAP_SKIPPED_EXCESSIVE_FALLBACK
        should_execute = False
    elif not sufficient:
        status = SAP_SKIPPED_INSUFFICIENT_REFERENCE
        should_execute = False
    else:
        status = SAP_EXECUTED
        should_execute = True
    return SAPGateDecision(
        should_execute=should_execute,
        status=status,
        current_class_coverage=current_class_coverage,
        current_classes_with_five_main=current_classes_with_five_main,
        seen_class_coverage=seen_class_coverage,
        reference_count=reference_count,
        fallback_fraction=fallback_fraction,
    )


def run_sap_projection_transaction(
    source_model,
    batch_factory,
    *,
    total_images: int,
    max_patches: int,
    scale: float,
    seed: int,
    dry_run: bool,
    diagnostic_batch_factories: dict[str, Callable] | None = None,
    seen_classes: int | None = None,
    test_task_batch_factories: Sequence[Callable] | None = None,
    enforce_candidate_safety: bool = False,
    required_replay_task_names: set[str] | None = None,
) -> SAPProjectionTransaction:
    """Project a candidate completely before optionally committing it to the source."""
    source_state = copy.deepcopy(source_model.state_dict())
    candidate = copy.deepcopy(source_model)
    try:
        layer_stats = project_resnet18_from_reference_batches(
            source_model,
            candidate,
            batch_factory,
            total_images=total_images,
            max_patches=max_patches,
            scale=scale,
            seed=seed,
        )
        if any(stats.relative_weight_delta <= 0 for stats in layer_stats.values()):
            raise ValueError('SAP produced a zero weight delta for at least one target layer')
        diagnostic_batch_factories = diagnostic_batch_factories or {}
        test_task_batch_factories = test_task_batch_factories or []
        if (diagnostic_batch_factories or test_task_batch_factories) and (
            seen_classes is None or seen_classes <= 0
        ):
            raise ValueError('seen_classes must be positive when SAP diagnostics are requested')
        comparisons = {
            name: compare_models_on_labeled_batches(
                source_model,
                candidate,
                factory,
                seen_classes=seen_classes,
            )
            for name, factory in diagnostic_batch_factories.items()
        }
        task_accuracy_comparisons = [
            compare_models_task_accuracy(
                source_model,
                candidate,
                factory,
                seen_classes=seen_classes,
                task_id=task_id,
            )
            for task_id, factory in enumerate(test_task_batch_factories)
        ]
        if enforce_candidate_safety:
            candidate_decision = assess_sap_candidate(
                layer_stats,
                comparisons,
                required_replay_task_names=required_replay_task_names,
            )
        else:
            candidate_decision = SAPCandidateDecision(
                accepted=True,
                rejection_reasons=(),
                minimum_weight_norm_ratio=min(
                    stats.weight_norm_ratio for stats in layer_stats.values()
                ),
                accuracy_deltas={},
            )
        target_weights = {f'{name}.weight' for name in RESNET18_LATE_STAGE_CONVS}
        candidate_state = candidate.state_dict()
        non_target_deltas = [
            (candidate_state[name] - value).abs().max().item()
            for name, value in source_state.items()
            if name not in target_weights
        ]
        max_non_target_state_delta = max(non_target_deltas, default=0.0)
        if max_non_target_state_delta != 0:
            raise ValueError('SAP changed at least one non-target parameter or buffer')
        if not dry_run and candidate_decision.accepted:
            source_model.load_state_dict(candidate.state_dict())
    except Exception:
        source_model.load_state_dict(source_state)
        raise
    finally:
        del candidate
    return SAPProjectionTransaction(
        committed=not dry_run and candidate_decision.accepted,
        accepted=candidate_decision.accepted,
        rejection_reasons=candidate_decision.rejection_reasons,
        minimum_weight_norm_ratio=candidate_decision.minimum_weight_norm_ratio,
        training_side_accuracy_deltas=candidate_decision.accuracy_deltas,
        layer_stats=dict(layer_stats),
        comparisons=comparisons,
        task_accuracy_comparisons=task_accuracy_comparisons,
        max_non_target_state_delta=max_non_target_state_delta,
    )


def compare_models_on_labeled_batches(
    before_model,
    after_model,
    batch_factory,
    *,
    seen_classes: int,
) -> SAPModelComparison:
    """Compare two models and stratify samples by pre-SAP CE loss tertile."""
    before_losses, after_losses, labels_all = [], [], []
    before_predictions, after_predictions = [], []
    logits_deltas = []
    before_states = {module: module.training for module in before_model.modules()}
    after_states = {module: module.training for module in after_model.modules()}
    try:
        before_model.eval()
        after_model.eval()
        with torch.no_grad():
            for inputs, labels in batch_factory():
                before_logits = before_model(inputs)[:, :seen_classes]
                after_logits = after_model(inputs)[:, :seen_classes]
                labels = labels.to(before_logits.device)
                before_losses.append(F.cross_entropy(before_logits, labels, reduction='none').cpu())
                after_losses.append(F.cross_entropy(after_logits, labels, reduction='none').cpu())
                labels_all.append(labels.cpu())
                before_predictions.append(before_logits.argmax(dim=1).cpu())
                after_predictions.append(after_logits.argmax(dim=1).cpu())
                logits_deltas.append((after_logits - before_logits).abs().cpu())
    finally:
        for module, state in before_states.items():
            module.training = state
        for module, state in after_states.items():
            module.training = state
    if not before_losses:
        raise ValueError('no labeled samples were provided for SAP diagnostics')

    before_loss = torch.cat(before_losses)
    after_loss = torch.cat(after_losses)
    labels = torch.cat(labels_all)
    prediction_before = torch.cat(before_predictions)
    prediction_after = torch.cat(after_predictions)
    logits_delta = torch.cat(logits_deltas)
    ordered_indices = torch.argsort(before_loss, stable=True)
    index_groups = torch.tensor_split(ordered_indices, 3)
    loss_tertiles = {}
    for name, indices in zip(('low', 'mid', 'high'), index_groups):
        before_group_loss = before_loss[indices]
        after_group_loss = after_loss[indices]
        before_correct = prediction_before[indices] == labels[indices]
        after_correct = prediction_after[indices] == labels[indices]
        loss_tertiles[name] = SAPLossGroupComparison(
            sample_count=len(indices),
            mean_loss_before=before_group_loss.mean().item(),
            mean_loss_after=after_group_loss.mean().item(),
            mean_loss_delta=(after_group_loss - before_group_loss).mean().item(),
            accuracy_before=before_correct.float().mean().item(),
            accuracy_after=after_correct.float().mean().item(),
            accuracy_delta=(after_correct.float().mean() - before_correct.float().mean()).item(),
            prediction_flip_rate=(prediction_before[indices] != prediction_after[indices]).float().mean().item(),
        )
    return SAPModelComparison(
        sample_count=len(labels),
        mean_abs_logits_delta=logits_delta.mean().item(),
        max_abs_logits_delta=logits_delta.max().item(),
        prediction_flip_rate=(prediction_before != prediction_after).float().mean().item(),
        loss_tertiles=loss_tertiles,
    )


def compare_models_task_accuracy(
    before_model,
    after_model,
    batch_factory,
    *,
    seen_classes: int,
    task_id: int,
) -> SAPTaskAccuracyComparison:
    """Compare Class-IL accuracy for one test task without selecting on it."""
    correct_before = 0
    correct_after = 0
    sample_count = 0
    before_states = {module: module.training for module in before_model.modules()}
    after_states = {module: module.training for module in after_model.modules()}
    try:
        before_model.eval()
        after_model.eval()
        with torch.no_grad():
            for inputs, labels in batch_factory():
                labels = labels.to(inputs.device)
                prediction_before = before_model(inputs)[:, :seen_classes].argmax(dim=1)
                prediction_after = after_model(inputs)[:, :seen_classes].argmax(dim=1)
                correct_before += (prediction_before == labels).sum().item()
                correct_after += (prediction_after == labels).sum().item()
                sample_count += labels.numel()
    finally:
        for module, state in before_states.items():
            module.training = state
        for module, state in after_states.items():
            module.training = state
    if sample_count == 0:
        raise ValueError('test task diagnostic received no samples')
    accuracy_before = correct_before / sample_count
    accuracy_after = correct_after / sample_count
    return SAPTaskAccuracyComparison(
        task_id=int(task_id),
        sample_count=sample_count,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        accuracy_delta=accuracy_after - accuracy_before,
    )
