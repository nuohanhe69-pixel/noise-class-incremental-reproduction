"""Validate SAP scoring and trusted references with real noisy CIFAR-10 data."""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backbone.ResNetBlock import resnet18
from utils.sap_reference import SAPReferenceMemory, score_seen_class_samples, select_task_references


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1)
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2615)).reshape(1, 3, 1, 1)


def _load_backbone(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_state = checkpoint['model']
    backbone_state = {
        name.removeprefix('net.'): value
        for name, value in checkpoint_state.items()
        if name.startswith('net.')
    }
    model = resnet18(num_classes=10)
    model.load_state_dict(backbone_state, strict=True)
    return model


def validate(data_root: Path, noisy_targets_path: Path, checkpoint_path: Path) -> None:
    from torchvision.datasets import CIFAR10

    clean_dataset = CIFAR10(root=data_root, train=True, download=False)
    with noisy_targets_path.open('rb') as handle:
        noisy_targets = np.asarray(pickle.load(handle), dtype=np.int64)
    clean_targets = np.asarray(clean_dataset.targets, dtype=np.int64)
    if len(clean_dataset) != 50_000 or noisy_targets.shape != clean_targets.shape:
        raise RuntimeError('CIFAR-10 data or noisy-label cache is incomplete')
    actual_noise_rate = float((noisy_targets != clean_targets).mean())
    if not np.isclose(actual_noise_rate, 0.4):
        raise RuntimeError(f'expected 40% noise, got {actual_noise_rate:.6f}')

    task_mask = np.isin(noisy_targets, [8, 9])
    task_sample_ids = np.flatnonzero(task_mask)
    task_images_uint8 = torch.from_numpy(clean_dataset.data[task_mask]).permute(0, 3, 1, 2).contiguous()
    task_observed_labels = torch.from_numpy(noisy_targets[task_mask]).long()
    task_true_labels = torch.from_numpy(clean_targets[task_mask]).long()
    normalized_images = task_images_uint8.float().div(255)
    normalized_images = (normalized_images - CIFAR10_MEAN) / CIFAR10_STD

    scoring_loader = DataLoader(
        TensorDataset(normalized_images, task_observed_labels),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    model = _load_backbone(checkpoint_path)
    scores = score_seen_class_samples(model, scoring_loader, seen_classes=10)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        references, reports = select_task_references(
            images=task_images_uint8,
            observed_labels=task_observed_labels,
            sample_ids=torch.from_numpy(task_sample_ids),
            source_task_id=4,
            losses=scores.losses,
            predictions=scores.predictions,
            confidences=scores.confidences,
            class_quota=15,
            seed=0,
        )
    memory = SAPReferenceMemory()
    memory.add_task(4, references)
    restored = SAPReferenceMemory.deserialize(memory.serialize())
    if len(restored) != len(memory) or len(restored) == 0:
        raise RuntimeError('reference memory did not survive serialization')

    selected_positions = {int(sample_id): index for index, sample_id in enumerate(task_sample_ids)}
    selected_true_labels = torch.tensor([
        int(task_true_labels[selected_positions[reference.sample_id]]) for reference in references
    ])
    selected_observed_labels = torch.tensor([reference.observed_label for reference in references])
    purity = float((selected_true_labels == selected_observed_labels).float().mean())
    fallback_count = sum(reference.selection_source != 'GMM_MAIN' for reference in references)

    for report in reports:
        print(
            'CLASS_REPORT', f'label={report.observed_label}', f'samples={report.sample_count}',
            f'valid_fits={report.valid_gmm_fits}', f'main_candidates={report.main_candidate_count}',
            f'selected_main={report.selected_main_count}', f'fallback={report.fallback_count}',
        )
    print(
        'REAL_CIFAR10_STAGE3_OK', f'task_images={len(task_sample_ids)}',
        f'noise_rate={actual_noise_rate:.4f}', f'references={len(references)}',
        f'fallback={fallback_count}', f'diagnostic_purity={purity:.4f}',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/CIFAR10'))
    parser.add_argument(
        '--noisy-targets', type=Path,
        default=Path('data/noisy_labels/seq-cifar10/symmetric/40/0/noisy_targets'),
    )
    parser.add_argument('--checkpoint', type=Path, required=True)
    arguments = parser.parse_args()
    validate(arguments.data_root, arguments.noisy_targets, arguments.checkpoint)
