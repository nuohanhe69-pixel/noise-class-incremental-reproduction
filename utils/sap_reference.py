"""Trusted-reference scoring, robust GMM selection, and persistent storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import Tensor, nn


GMM_MAIN = 'GMM_MAIN'
GMM_FALLBACK_LOW_LOSS = 'GMM_FALLBACK_LOW_LOSS'
MULTISTAGE_PROMOTED = 'MULTISTAGE_PROMOTED'
TRUSTED_SELECTION_SOURCES = frozenset((GMM_MAIN, MULTISTAGE_PROMOTED))


@dataclass(frozen=True)
class SAPScoreBatch:
    losses: Tensor
    predictions: Tensor
    confidences: Tensor


@dataclass(frozen=True)
class RobustGMMResult:
    probability_mean: np.ndarray
    probability_std: np.ndarray
    clean_votes: np.ndarray
    main_mask: np.ndarray
    valid_fits: int
    acceptance_tier: str
    rejection_reasons: tuple[str, ...]
    attempted_fits: int
    converged_fits: int
    separation_median: float
    separation_std: float


@dataclass(frozen=True)
class GMMStabilityDecision:
    accepted: bool
    acceptance_tier: str
    rejection_reasons: tuple[str, ...]
    attempted_fits: int
    converged_fits: int
    separation_median: float
    separation_std: float
    minimum_component_weight: float


@dataclass(frozen=True)
class SAPClassSelectionReport:
    observed_label: int
    sample_count: int
    valid_gmm_fits: int
    main_candidate_count: int
    selected_main_count: int
    fallback_count: int
    gmm_acceptance_tier: str
    gmm_rejection_reasons: tuple[str, ...]
    separation_median: float
    separation_std: float


def assess_gmm_stability(
    *,
    separations: Sequence[float],
    component_min_weights: Sequence[float],
    converged_fits: int,
    attempted_fits: int,
    required_fits: int = 4,
    min_component_weight: float = 0.05,
    strict_separation: float = 1.0,
    boundary_separation: float = 0.9,
    max_boundary_separation_std: float = 0.03,
) -> GMMStabilityDecision:
    """Classify a bootstrap GMM ensemble using strict and stable-boundary tiers."""
    separation_array = np.asarray(separations, dtype=np.float64)
    weight_array = np.asarray(component_min_weights, dtype=np.float64)
    finite = (
        separation_array.size > 0
        and weight_array.size == separation_array.size
        and np.isfinite(separation_array).all()
        and np.isfinite(weight_array).all()
    )
    reasons = []
    if converged_fits < required_fits:
        reasons.append('INSUFFICIENT_CONVERGED_FITS')
    if not finite:
        reasons.append('NON_FINITE_FIT_STATISTICS')
        separation_median = 0.0
        separation_std = float('inf')
        minimum_weight = 0.0
    else:
        separation_median = float(np.median(separation_array))
        separation_std = float(np.std(separation_array))
        minimum_weight = float(weight_array.min())
        if minimum_weight < min_component_weight:
            reasons.append('COMPONENT_WEIGHT_COLLAPSE')

    acceptance_tier = 'REJECTED'
    if not reasons and separation_median >= strict_separation:
        acceptance_tier = 'STRICT'
    elif not reasons and separation_median >= boundary_separation:
        if separation_std <= max_boundary_separation_std:
            acceptance_tier = 'STABLE_BOUNDARY'
        else:
            reasons.append('SEPARATION_UNSTABLE')
    elif finite and separation_median < boundary_separation:
        reasons.append('SEPARATION_BELOW_FLOOR')

    return GMMStabilityDecision(
        accepted=acceptance_tier != 'REJECTED',
        acceptance_tier=acceptance_tier,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        attempted_fits=int(attempted_fits),
        converged_fits=int(converged_fits),
        separation_median=separation_median,
        separation_std=separation_std,
        minimum_component_weight=minimum_weight,
    )


@dataclass(frozen=True, eq=False)
class SAPReference:
    image: Tensor
    observed_label: int
    source_task_id: int
    sample_id: int
    selection_loss: float
    clean_probability_mean: float
    clean_probability_std: float
    clean_votes: int
    final_seen_class_loss: float
    final_prediction: int
    final_confidence: float
    selection_source: str
    promotion_evidence: dict | None = None
    promoted_at_task_id: int | None = None

    def clone_for_task(self, source_task_id: int, sample_id: int) -> 'SAPReference':
        return replace(self, source_task_id=int(source_task_id), sample_id=int(sample_id))


def split_trusted_and_pending(
    references: Sequence[SAPReference],
) -> tuple[list[SAPReference], list[SAPReference]]:
    """Partition selected references without changing their evidence records."""
    trusted = [
        reference for reference in references
        if reference.selection_source in TRUSTED_SELECTION_SOURCES
    ]
    pending = [
        reference for reference in references
        if reference.selection_source not in TRUSTED_SELECTION_SOURCES
    ]
    return trusted, pending


@dataclass(frozen=True)
class SAPTrajectorySnapshot:
    epoch: int
    sample_ids: Tensor
    observed_labels: Tensor
    losses: Tensor
    predictions: Tensor
    confidences: Tensor
    class_loss_quantiles: Tensor
    ogc_high_confidence: Tensor
    abs_insertion_eligible: Tensor
    ogc_probability_threshold: float


def build_trajectory_snapshot(
    *,
    epoch: int,
    sample_ids: Tensor,
    observed_labels: Tensor,
    scores: SAPScoreBatch,
    ogc_probability_threshold: float,
    ogc_low_conf_weight: float,
    ogc_buffer_penalty_coeff: float,
    alpha_sample_insertion: float,
) -> SAPTrajectorySnapshot:
    """Convert a full-task scoring pass into auditable OGC/AER/ABS evidence."""
    sample_ids = sample_ids.detach().cpu().long().reshape(-1)
    observed_labels = observed_labels.detach().cpu().long().reshape(-1)
    losses = scores.losses.detach().cpu().float().reshape(-1)
    predictions = scores.predictions.detach().cpu().long().reshape(-1)
    confidences = scores.confidences.detach().cpu().float().reshape(-1)
    count = sample_ids.numel()
    if any(field.numel() != count for field in (observed_labels, losses, predictions, confidences)):
        raise ValueError('trajectory scoring inputs are not aligned')
    if not 0 <= alpha_sample_insertion <= 1:
        raise ValueError('alpha_sample_insertion must lie in [0, 1]')

    class_loss_quantiles = torch.empty(count, dtype=torch.float32)
    ogc_high_confidence = confidences > float(ogc_probability_threshold)
    ogc_weights = torch.where(
        ogc_high_confidence,
        torch.ones_like(confidences),
        torch.full_like(confidences, float(ogc_low_conf_weight)),
    )
    insertion_scores = losses * (float(ogc_buffer_penalty_coeff) - ogc_weights)
    abs_insertion_eligible = torch.zeros(count, dtype=torch.bool)
    for observed_label in observed_labels.unique(sorted=True):
        class_indices = torch.where(observed_labels == observed_label)[0]
        order = torch.argsort(losses[class_indices], stable=True)
        ranks = torch.empty(len(class_indices), dtype=torch.float32)
        denominator = max(len(class_indices) - 1, 1)
        ranks[order] = torch.arange(len(class_indices), dtype=torch.float32) / denominator
        class_loss_quantiles[class_indices] = ranks

        eligible_count = round((1 - alpha_sample_insertion) * len(class_indices))
        if eligible_count > 0:
            insertion_order = torch.argsort(insertion_scores[class_indices], stable=True)
            abs_insertion_eligible[class_indices[insertion_order[:eligible_count]]] = True

    return SAPTrajectorySnapshot(
        epoch=int(epoch),
        sample_ids=sample_ids,
        observed_labels=observed_labels,
        losses=losses,
        predictions=predictions,
        confidences=confidences,
        class_loss_quantiles=class_loss_quantiles,
        ogc_high_confidence=ogc_high_confidence,
        abs_insertion_eligible=abs_insertion_eligible,
        ogc_probability_threshold=float(ogc_probability_threshold),
    )


def promote_pending_references(
    pending: Sequence[SAPReference],
    snapshots: Sequence[SAPTrajectorySnapshot],
    *,
    promoted_at_task_id: int,
    required_epochs: tuple[int, ...] = (35, 45, 50),
) -> tuple[list[SAPReference], list[SAPReference], list[dict]]:
    """Promote fallback only when three independent training snapshots agree."""
    snapshots_by_epoch = {int(snapshot.epoch): snapshot for snapshot in snapshots}
    if set(snapshots_by_epoch) != set(required_epochs):
        raise ValueError(f'pending promotion requires snapshots at epochs {required_epochs}')
    indexed_snapshots = []
    for epoch in required_epochs:
        snapshot = snapshots_by_epoch[epoch]
        fields = (
            snapshot.observed_labels,
            snapshot.losses,
            snapshot.predictions,
            snapshot.confidences,
            snapshot.class_loss_quantiles,
            snapshot.ogc_high_confidence,
            snapshot.abs_insertion_eligible,
        )
        if any(field.numel() != snapshot.sample_ids.numel() for field in fields):
            raise ValueError(f'trajectory snapshot at epoch {epoch} is not aligned')
        index_by_sample_id = {
            int(sample_id): index for index, sample_id in enumerate(snapshot.sample_ids.tolist())
        }
        indexed_snapshots.append((snapshot, index_by_sample_id))

    promoted, remaining, reports = [], [], []
    for reference in pending:
        evidence_rows = []
        missing = False
        for snapshot, index_by_sample_id in indexed_snapshots:
            if reference.sample_id not in index_by_sample_id:
                missing = True
                break
            index = index_by_sample_id[reference.sample_id]
            if int(snapshot.observed_labels[index]) != reference.observed_label:
                raise ValueError('trajectory observed label changed for a pending sample')
            evidence_rows.append({
                'epoch': int(snapshot.epoch),
                'loss': float(snapshot.losses[index]),
                'prediction': int(snapshot.predictions[index]),
                'confidence': float(snapshot.confidences[index]),
                'class_loss_quantile': float(snapshot.class_loss_quantiles[index]),
                'ogc_high_confidence': bool(snapshot.ogc_high_confidence[index]),
                'abs_insertion_eligible': bool(snapshot.abs_insertion_eligible[index]),
                'ogc_probability_threshold': float(snapshot.ogc_probability_threshold),
            })

        rejection_reasons = []
        if missing:
            rejection_reasons.append('MISSING_TRAJECTORY')
        else:
            prediction_agreements = sum(
                row['prediction'] == reference.observed_label for row in evidence_rows
            )
            low_loss_votes = sum(row['class_loss_quantile'] <= 0.5 for row in evidence_rows)
            ogc_high_votes = sum(row['ogc_high_confidence'] for row in evidence_rows)
            abs_votes = sum(row['abs_insertion_eligible'] for row in evidence_rows)
            losses = np.asarray([row['loss'] for row in evidence_rows], dtype=np.float64)
            previous_minimum_loss = float(losses[:-1].min())
            final_loss_rebound = float(losses[-1] - previous_minimum_loss)
            if prediction_agreements != len(required_epochs):
                rejection_reasons.append('PREDICTION_INCONSISTENT')
            if low_loss_votes < 2:
                rejection_reasons.append('INSUFFICIENT_LOW_LOSS_VOTES')
            if ogc_high_votes < 2:
                rejection_reasons.append('INSUFFICIENT_OGC_HIGH_CONFIDENCE')
            if abs_votes < 1:
                rejection_reasons.append('NO_ABS_INSERTION_EVIDENCE')
            if final_loss_rebound > 0.1 and losses[-1] > 2 * max(previous_minimum_loss, 1e-8):
                rejection_reasons.append('LATE_LOSS_REBOUND')

        evidence = {
            'snapshots': evidence_rows,
            'rejection_reasons': tuple(rejection_reasons),
        }
        if not rejection_reasons:
            promoted.append(replace(
                reference,
                selection_source=MULTISTAGE_PROMOTED,
                promotion_evidence=evidence,
                promoted_at_task_id=int(promoted_at_task_id),
            ))
        else:
            remaining.append(reference)
        reports.append({
            'sample_id': reference.sample_id,
            'observed_label': reference.observed_label,
            'promoted': not rejection_reasons,
            **evidence,
        })
    return promoted, remaining, reports


def score_seen_class_samples(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    seen_classes: int,
) -> SAPScoreBatch:
    """Score observed labels against learned classes only."""
    if seen_classes <= 0:
        raise ValueError('seen_classes must be positive')
    losses, predictions, confidences = [], [], []
    training_states = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            for inputs, observed_labels in batches:
                logits = model(inputs)
                if logits.ndim != 2 or logits.shape[1] < seen_classes:
                    raise ValueError('model logits do not contain all seen classes')
                seen_logits = logits[:, :seen_classes]
                observed_labels = observed_labels.to(seen_logits.device)
                if observed_labels.numel() and observed_labels.max().item() >= seen_classes:
                    raise ValueError('observed label lies outside the seen-class range')
                probabilities = F.softmax(seen_logits, dim=1)
                losses.append(F.cross_entropy(seen_logits, observed_labels, reduction='none').cpu())
                confidences.append(
                    probabilities[torch.arange(observed_labels.shape[0], device=seen_logits.device), observed_labels].cpu()
                )
                predictions.append(seen_logits.argmax(dim=1).cpu())
    finally:
        for module, was_training in training_states.items():
            module.training = was_training
    if not losses:
        raise ValueError('no samples were provided for SAP scoring')
    return SAPScoreBatch(
        losses=torch.cat(losses),
        predictions=torch.cat(predictions),
        confidences=torch.cat(confidences),
    )


def fit_class_robust_gmm(
    losses: Sequence[float] | np.ndarray,
    *,
    seed: int,
    bootstraps: int = 5,
    min_samples: int = 100,
    min_loss_std: float = 0.02,
    min_component_weight: float = 0.05,
    min_separation: float = 1.0,
    boundary_separation: float = 0.9,
    max_boundary_separation_std: float = 0.03,
) -> RobustGMMResult:
    """Fit bootstrapped two-component GMMs and identify stable low-loss samples."""
    raw = np.asarray(losses, dtype=np.float64).reshape(-1)
    count = raw.size
    empty_probabilities = np.zeros(count, dtype=np.float64)
    empty_votes = np.zeros(count, dtype=np.int64)
    empty_mask = np.zeros(count, dtype=bool)

    def rejected(reason: str, *, attempted_fits: int = 0, converged_fits: int = 0,
                 separation_median: float = 0.0, separation_std: float = 0.0) -> RobustGMMResult:
        return RobustGMMResult(
            empty_probabilities,
            empty_probabilities.copy(),
            empty_votes,
            empty_mask,
            0,
            'REJECTED',
            (reason,),
            attempted_fits,
            converged_fits,
            separation_median,
            separation_std,
        )

    if count < min_samples:
        return rejected('PRECHECK_INSUFFICIENT_SAMPLES')
    if not np.isfinite(raw).all():
        return rejected('PRECHECK_NON_FINITE_LOSS')
    if raw.std() < min_loss_std:
        return rejected('PRECHECK_LOW_LOSS_VARIANCE')

    low, high = np.percentile(raw, [1.0, 99.0])
    if not high > low:
        return rejected('PRECHECK_ZERO_LOSS_RANGE')
    scaled = np.clip((raw - low) / (high - low), 0.0, 1.0).reshape(-1, 1)
    rng = np.random.RandomState(seed)
    posterior_runs = []
    vote_runs = []
    separations = []
    minimum_weights = []
    converged_fits = 0
    fit_rejection_reasons = []

    for bootstrap_index in range(bootstraps):
        sampled_indices = rng.randint(0, count, size=count)
        model = GaussianMixture(
            n_components=2,
            n_init=10,
            reg_covar=1e-4,
            init_params='random_from_data',
            random_state=int(seed) + bootstrap_index,
        )
        try:
            model.fit(scaled[sampled_indices])
        except (ValueError, FloatingPointError):
            fit_rejection_reasons.append('FIT_EXCEPTION')
            continue
        if not model.converged_:
            fit_rejection_reasons.append('FIT_NOT_CONVERGED')
            continue
        converged_fits += 1
        means = model.means_.reshape(-1)
        weights = model.weights_.reshape(-1)
        variances = model.covariances_.reshape(2, -1).mean(axis=1)
        pooled_std = np.sqrt(max(float(variances.sum()), np.finfo(np.float64).eps))
        separation = abs(float(means[0] - means[1])) / pooled_std
        if not np.isfinite(means).all() or not np.isfinite(weights).all() or not np.isfinite(separation):
            fit_rejection_reasons.append('FIT_NON_FINITE')
            continue
        clean_component = int(np.argmin(means))
        posterior = model.predict_proba(scaled)[:, clean_component]
        posterior_runs.append(posterior)
        vote_runs.append(posterior >= 0.5)
        separations.append(separation)
        minimum_weights.append(float(weights.min()))

    decision = assess_gmm_stability(
        separations=separations,
        component_min_weights=minimum_weights,
        converged_fits=converged_fits,
        attempted_fits=bootstraps,
        min_component_weight=min_component_weight,
        strict_separation=min_separation,
        boundary_separation=boundary_separation,
        max_boundary_separation_std=max_boundary_separation_std,
    )
    if not decision.accepted:
        reasons = tuple(dict.fromkeys((*fit_rejection_reasons, *decision.rejection_reasons)))
        return RobustGMMResult(
            empty_probabilities,
            empty_probabilities.copy(),
            empty_votes,
            empty_mask,
            0,
            decision.acceptance_tier,
            reasons,
            decision.attempted_fits,
            decision.converged_fits,
            decision.separation_median,
            decision.separation_std,
        )
    valid_fits = len(posterior_runs)
    probabilities = np.stack(posterior_runs)
    votes = np.stack(vote_runs).sum(axis=0)
    probability_mean = probabilities.mean(axis=0)
    probability_std = probabilities.std(axis=0)
    main_mask = (
        (valid_fits >= 4)
        & (probability_mean >= 0.9)
        & (votes >= 4)
        & (probability_std <= 0.1)
    )
    sample_reasons = list(fit_rejection_reasons)
    if not main_mask.any():
        sample_reasons.append('NO_SAMPLE_PASSED_POSTERIOR_GATE')
    return RobustGMMResult(
        probability_mean,
        probability_std,
        votes,
        main_mask,
        valid_fits,
        decision.acceptance_tier,
        tuple(dict.fromkeys(sample_reasons)),
        decision.attempted_fits,
        decision.converged_fits,
        decision.separation_median,
        decision.separation_std,
    )


def _ensure_uint8_images(images: Tensor) -> Tensor:
    if images.ndim != 4:
        raise ValueError('reference images must be NCHW')
    if images.dtype == torch.uint8:
        return images.detach().cpu().clone()
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise TypeError('reference images must be uint8 or finite floating-point tensors')
    if images.min().item() < 0 or images.max().item() > 1:
        raise ValueError('floating-point reference images must lie in [0, 1]')
    return images.detach().cpu().mul(255).round().to(torch.uint8)


def select_task_references(
    *,
    images: Tensor,
    observed_labels: Tensor,
    sample_ids: Tensor,
    source_task_id: int,
    losses: Tensor,
    predictions: Tensor,
    confidences: Tensor,
    class_quota: int,
    seed: int,
) -> tuple[list[SAPReference], list[SAPClassSelectionReport]]:
    """Select per-observed-class trusted references with conservative fallback."""
    if class_quota <= 0:
        raise ValueError('class_quota must be positive')
    cpu_images = _ensure_uint8_images(images)
    tensors = [observed_labels, sample_ids, losses, predictions, confidences]
    count = cpu_images.shape[0]
    if any(tensor.numel() != count for tensor in tensors):
        raise ValueError('SAP selection inputs must have the same number of samples')
    observed_labels = observed_labels.detach().cpu().long()
    sample_ids = sample_ids.detach().cpu().long()
    losses = losses.detach().cpu().float()
    predictions = predictions.detach().cpu().long()
    confidences = confidences.detach().cpu().float()

    selected = []
    reports = []
    for class_offset, observed_label_tensor in enumerate(observed_labels.unique(sorted=True)):
        observed_label = int(observed_label_tensor.item())
        class_indices = torch.where(observed_labels == observed_label)[0]
        class_losses = losses[class_indices]
        gmm = fit_class_robust_gmm(class_losses.numpy(), seed=int(seed) + class_offset)
        prediction_agrees = predictions[class_indices].numpy() == observed_label
        main_local_indices = np.flatnonzero(gmm.main_mask & prediction_agrees)
        main_local_indices = sorted(main_local_indices, key=lambda index: (float(class_losses[index]), int(sample_ids[class_indices[index]])))
        chosen_main = main_local_indices[:class_quota]
        chosen = [(index, GMM_MAIN) for index in chosen_main]

        if gmm.valid_fits < 4 or len(main_local_indices) < 5:
            chosen_set = set(chosen_main)
            remaining = [
                index for index in range(len(class_indices))
                if index not in chosen_set and int(predictions[class_indices[index]]) == observed_label
            ]
            remaining.sort(key=lambda index: (float(class_losses[index]), int(sample_ids[class_indices[index]])))
            top_up = max(0, min(5, class_quota) - len(chosen))
            chosen.extend((index, GMM_FALLBACK_LOW_LOSS) for index in remaining[:top_up])

        fallback_count = 0
        selected_main_count = 0
        for local_index, source in chosen:
            global_index = int(class_indices[local_index])
            if source == GMM_MAIN:
                selected_main_count += 1
            else:
                fallback_count += 1
            selected.append(SAPReference(
                image=cpu_images[global_index].clone(),
                observed_label=observed_label,
                source_task_id=int(source_task_id),
                sample_id=int(sample_ids[global_index]),
                selection_loss=float(losses[global_index]),
                clean_probability_mean=float(gmm.probability_mean[local_index]),
                clean_probability_std=float(gmm.probability_std[local_index]),
                clean_votes=int(gmm.clean_votes[local_index]),
                final_seen_class_loss=float(losses[global_index]),
                final_prediction=int(predictions[global_index]),
                final_confidence=float(confidences[global_index]),
                selection_source=source,
            ))
        reports.append(SAPClassSelectionReport(
            observed_label=observed_label,
            sample_count=len(class_indices),
            valid_gmm_fits=gmm.valid_fits,
            main_candidate_count=int(gmm.main_mask.sum()),
            selected_main_count=selected_main_count,
            fallback_count=fallback_count,
            gmm_acceptance_tier=gmm.acceptance_tier,
            gmm_rejection_reasons=gmm.rejection_reasons,
            separation_median=gmm.separation_median,
            separation_std=gmm.separation_std,
        ))
    return selected, reports


def _reference_to_state(reference: SAPReference) -> dict:
    return {
        'image': reference.image.detach().cpu().to(torch.uint8).clone(),
        'observed_label': reference.observed_label,
        'source_task_id': reference.source_task_id,
        'sample_id': reference.sample_id,
        'selection_loss': reference.selection_loss,
        'clean_probability_mean': reference.clean_probability_mean,
        'clean_probability_std': reference.clean_probability_std,
        'clean_votes': reference.clean_votes,
        'final_seen_class_loss': reference.final_seen_class_loss,
        'final_prediction': reference.final_prediction,
        'final_confidence': reference.final_confidence,
        'selection_source': reference.selection_source,
        'promotion_evidence': reference.promotion_evidence,
        'promoted_at_task_id': reference.promoted_at_task_id,
    }


class SAPReferenceMemory:
    """Task-partitioned reference memory whose prior task members are immutable."""

    def __init__(self) -> None:
        self._tasks: dict[int, list[SAPReference]] = {}

    def add_task(self, task_id: int, references: Sequence[SAPReference]) -> None:
        task_id = int(task_id)
        if task_id in self._tasks:
            raise ValueError(f'SAP references for task {task_id} already exist')
        copied = []
        seen_sample_ids = set()
        for reference in references:
            if reference.source_task_id != task_id:
                raise ValueError('reference source_task_id does not match task_id')
            if reference.sample_id in seen_sample_ids:
                raise ValueError('duplicate sample_id within SAP task references')
            seen_sample_ids.add(reference.sample_id)
            copied.append(replace(reference, image=reference.image.detach().cpu().to(torch.uint8).clone()))
        self._tasks[task_id] = copied

    def items(self) -> list[SAPReference]:
        return [reference for task_id in sorted(self._tasks) for reference in self._tasks[task_id]]

    def __len__(self) -> int:
        return sum(len(references) for references in self._tasks.values())

    def serialize(self) -> dict:
        return {
            'version': 1,
            'tasks': {
                str(task_id): [_reference_to_state(reference) for reference in references]
                for task_id, references in sorted(self._tasks.items())
            },
        }

    @classmethod
    def deserialize(cls, state: dict) -> 'SAPReferenceMemory':
        if state.get('version') != 1 or not isinstance(state.get('tasks'), dict):
            raise ValueError('unsupported SAP reference checkpoint state')
        memory = cls()
        for task_id_text, reference_states in sorted(state['tasks'].items(), key=lambda item: int(item[0])):
            references = [SAPReference(**reference_state) for reference_state in reference_states]
            memory.add_task(int(task_id_text), references)
        return memory


def serialize_reference_memories(
    trusted: SAPReferenceMemory,
    pending: SAPReferenceMemory,
) -> dict:
    return {
        'version': 2,
        'trusted': trusted.serialize(),
        'pending': pending.serialize(),
    }


def deserialize_reference_memories(
    state: dict,
) -> tuple[SAPReferenceMemory, SAPReferenceMemory, bool]:
    """Load v2 stores or explicitly split a legacy v1 mixed memory."""
    if state.get('version') == 2:
        return (
            SAPReferenceMemory.deserialize(state['trusted']),
            SAPReferenceMemory.deserialize(state['pending']),
            False,
        )
    if state.get('version') != 1:
        raise ValueError('unsupported SAP reference collections state')

    legacy = SAPReferenceMemory.deserialize(state)
    trusted = SAPReferenceMemory()
    pending = SAPReferenceMemory()
    task_ids = sorted({reference.source_task_id for reference in legacy.items()})
    for task_id in task_ids:
        task_references = [
            reference for reference in legacy.items() if reference.source_task_id == task_id
        ]
        task_trusted, task_pending = split_trusted_and_pending(task_references)
        trusted.add_task(task_id, task_trusted)
        pending.add_task(task_id, task_pending)
    return trusted, pending, True
