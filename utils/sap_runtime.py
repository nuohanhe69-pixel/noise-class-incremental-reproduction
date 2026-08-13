"""Runtime gates and data adapters for task-boundary SAP."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from utils.sap import SAPLayerProjectionStats, project_resnet18_from_reference_batches
from utils.sap_reference import GMM_MAIN, SAPClassSelectionReport, SAPReferenceMemory


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
    layer_stats: dict[str, SAPLayerProjectionStats]


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
    current_class_coverage = sum(report.selected_main_count + report.fallback_count > 0 for report in current_reports)
    current_classes_with_five_main = sum(report.selected_main_count >= 5 for report in current_reports)
    references = memory.items()
    seen_class_coverage = len({reference.observed_label for reference in references})
    fallback_count = sum(reference.selection_source != GMM_MAIN for reference in references)
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
        if not dry_run:
            source_model.load_state_dict(candidate.state_dict())
    except Exception:
        source_model.load_state_dict(source_state)
        raise
    finally:
        del candidate
    return SAPProjectionTransaction(
        committed=not dry_run,
        layer_stats=dict(layer_stats),
    )
