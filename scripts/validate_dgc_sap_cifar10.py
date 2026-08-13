"""Validate DGC-SAP transactions and checkpoint state on real CIFAR-10 data."""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backbone.ResNetBlock import resnet18
from utils.sap import RESNET18_LATE_STAGE_CONVS
from utils.sap_reference import SAPReferenceMemory, score_seen_class_samples, select_task_references
from utils.sap_runtime import run_sap_projection_transaction


MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2615)).reshape(1, 3, 1, 1)


def _load_trained_backbone(checkpoint_path: Path):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    backbone_state = {
        name.removeprefix('net.'): value
        for name, value in saved['model'].items()
        if name.startswith('net.')
    }
    model = resnet18(num_classes=10)
    model.load_state_dict(backbone_state, strict=True)
    return model


def _normalized_batches(images: torch.Tensor, batch_size: int = 16):
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].float().div(255)
        yield (batch - MEAN) / STD


def _assert_equal_state(model, state):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def validate(data_root: Path, noisy_targets_path: Path, trained_checkpoint: Path, sap_checkpoint: Path) -> None:
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=data_root, train=True, download=False)
    with noisy_targets_path.open('rb') as handle:
        noisy_targets = np.asarray(pickle.load(handle), dtype=np.int64)
    mask = np.isin(noisy_targets, [8, 9])
    sample_ids = np.flatnonzero(mask)
    images = torch.from_numpy(dataset.data[mask]).permute(0, 3, 1, 2).contiguous()
    labels = torch.from_numpy(noisy_targets[mask]).long()
    normalized = torch.cat(list(_normalized_batches(images, batch_size=256)))

    trained_model = _load_trained_backbone(trained_checkpoint)
    scores = score_seen_class_samples(
        trained_model,
        DataLoader(TensorDataset(normalized, labels), batch_size=128, shuffle=False),
        seen_classes=10,
    )
    references, _ = select_task_references(
        images=images,
        observed_labels=labels,
        sample_ids=torch.from_numpy(sample_ids),
        source_task_id=4,
        losses=scores.losses,
        predictions=scores.predictions,
        confidences=scores.confidences,
        class_quota=15,
        seed=0,
    )
    if len(references) != 30:
        raise RuntimeError(f'expected 30 real references, got {len(references)}')
    reference_images = torch.stack([reference.image for reference in references])

    source = resnet18(num_classes=10, num_filters=2)
    before = copy.deepcopy(source.state_dict())
    dry = run_sap_projection_transaction(
        source,
        lambda: _normalized_batches(reference_images),
        total_images=len(reference_images),
        max_patches=32,
        scale=3000.0,
        seed=0,
        dry_run=True,
    )
    if dry.committed:
        raise RuntimeError('dry-run unexpectedly committed weights')
    _assert_equal_state(source, before)
    print('DRY_RUN_OK', f'references={len(reference_images)}', f'layers={len(dry.layer_stats)}')

    committed = run_sap_projection_transaction(
        source,
        lambda: _normalized_batches(reference_images),
        total_images=len(reference_images),
        max_patches=32,
        scale=3000.0,
        seed=0,
        dry_run=False,
    )
    target_weights = {f'{name}.weight' for name in RESNET18_LATE_STAGE_CONVS}
    for name, value in source.state_dict().items():
        if name in target_weights:
            if torch.equal(value, before[name]):
                raise RuntimeError(f'target layer was not committed: {name}')
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    print('COMMIT_OK', f'layers={len(committed.layer_stats)}')

    rollback_source = resnet18(num_classes=10, num_filters=2)
    rollback_before = copy.deepcopy(rollback_source.state_dict())
    try:
        run_sap_projection_transaction(
            rollback_source,
            lambda: [],
            total_images=len(reference_images),
            max_patches=32,
            scale=3000.0,
            seed=0,
            dry_run=False,
        )
    except ValueError:
        pass
    else:
        raise RuntimeError('expected the failing transaction to raise ValueError')
    _assert_equal_state(rollback_source, rollback_before)
    print('ROLLBACK_OK')

    saved = torch.load(sap_checkpoint, map_location='cpu', weights_only=False)
    sap_state = saved.get('sap_state')
    if not sap_state or not sap_state.get('history'):
        raise RuntimeError('DGC-SAP checkpoint is missing SAP state/history')
    restored = SAPReferenceMemory.deserialize(sap_state['reference_memory'])
    if len(restored) == 0:
        raise RuntimeError('DGC-SAP checkpoint restored an empty reference memory')
    print(
        'CHECKPOINT_OK', f'references={len(restored)}',
        f'last_status={sap_state["history"][-1]["status"]}',
    )
    print('REAL_CIFAR10_STAGE4_OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/CIFAR10'))
    parser.add_argument(
        '--noisy-targets', type=Path,
        default=Path('data/noisy_labels/seq-cifar10/symmetric/40/0/noisy_targets'),
    )
    parser.add_argument('--trained-checkpoint', type=Path, required=True)
    parser.add_argument('--sap-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    validate(args.data_root, args.noisy_targets, args.trained_checkpoint, args.sap_checkpoint)
