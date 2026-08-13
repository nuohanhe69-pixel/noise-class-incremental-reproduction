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


@dataclass(frozen=True)
class SAPClassSelectionReport:
    observed_label: int
    sample_count: int
    valid_gmm_fits: int
    main_candidate_count: int
    selected_main_count: int
    fallback_count: int


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

    def clone_for_task(self, source_task_id: int, sample_id: int) -> 'SAPReference':
        return replace(self, source_task_id=int(source_task_id), sample_id=int(sample_id))


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
) -> RobustGMMResult:
    """Fit bootstrapped two-component GMMs and identify stable low-loss samples."""
    raw = np.asarray(losses, dtype=np.float64).reshape(-1)
    count = raw.size
    empty_probabilities = np.zeros(count, dtype=np.float64)
    empty_votes = np.zeros(count, dtype=np.int64)
    if count < min_samples or not np.isfinite(raw).all() or raw.std() < min_loss_std:
        return RobustGMMResult(empty_probabilities, empty_probabilities.copy(), empty_votes, np.zeros(count, dtype=bool), 0)

    low, high = np.percentile(raw, [1.0, 99.0])
    if not high > low:
        return RobustGMMResult(empty_probabilities, empty_probabilities.copy(), empty_votes, np.zeros(count, dtype=bool), 0)
    scaled = np.clip((raw - low) / (high - low), 0.0, 1.0).reshape(-1, 1)
    rng = np.random.RandomState(seed)
    posterior_runs = []
    vote_runs = []

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
            continue
        means = model.means_.reshape(-1)
        weights = model.weights_.reshape(-1)
        variances = model.covariances_.reshape(2, -1).mean(axis=1)
        pooled_std = np.sqrt(max(float(variances.sum()), np.finfo(np.float64).eps))
        separation = abs(float(means[0] - means[1])) / pooled_std
        if (
            not model.converged_
            or float(weights.min()) < min_component_weight
            or separation < min_separation
        ):
            continue
        clean_component = int(np.argmin(means))
        posterior = model.predict_proba(scaled)[:, clean_component]
        posterior_runs.append(posterior)
        vote_runs.append(posterior >= 0.5)

    valid_fits = len(posterior_runs)
    if valid_fits == 0:
        return RobustGMMResult(empty_probabilities, empty_probabilities.copy(), empty_votes, np.zeros(count, dtype=bool), 0)
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
    return RobustGMMResult(probability_mean, probability_std, votes, main_mask, valid_fits)


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
        main_local_indices = np.flatnonzero(gmm.main_mask)
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
