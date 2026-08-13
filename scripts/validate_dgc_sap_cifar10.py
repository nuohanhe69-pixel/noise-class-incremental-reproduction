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
    test_dataset = CIFAR10(root=data_root, train=False, download=False)
    with noisy_targets_path.open('rb') as handle:
        noisy_targets = np.asarray(pickle.load(handle), dtype=np.int64)
    mask = np.isin(noisy_targets, [8, 9])
    sample_ids = np.flatnonzero(mask)
    images = torch.from_numpy(dataset.data[mask]).permute(0, 3, 1, 2).contiguous()
    labels = torch.from_numpy(noisy_targets[mask]).long()
    normalized = torch.cat(list(_normalized_batches(images, batch_size=256)))

    trained_saved = torch.load(trained_checkpoint, map_location='cpu', weights_only=False)
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
    reference_labels = torch.tensor([reference.observed_label for reference in references])
    buffer_images = trained_saved['buffer']['examples']
    buffer_labels = trained_saved['buffer']['labels']

    def labeled_batches(batch_images, batch_labels, batch_size=32):
        for start in range(0, len(batch_images), batch_size):
            normalized_batch = list(_normalized_batches(
                batch_images[start:start + batch_size], batch_size=batch_size,
            ))[0]
            yield normalized_batch, batch_labels[start:start + batch_size]

    test_images = torch.from_numpy(test_dataset.data).permute(0, 3, 1, 2).contiguous()
    test_labels = torch.tensor(test_dataset.targets)
    test_task_factories = []
    for task_id in range(5):
        task_mask = (test_labels >= task_id * 2) & (test_labels < (task_id + 1) * 2)
        task_images = test_images[task_mask]
        task_labels = test_labels[task_mask]
        test_task_factories.append(
            lambda images=task_images, labels=task_labels: labeled_batches(images, labels, batch_size=128)
        )

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
        diagnostic_batch_factories={
            'reference': lambda: labeled_batches(reference_images, reference_labels),
            'replay_buffer': lambda: labeled_batches(buffer_images, buffer_labels),
        },
        seen_classes=10,
        test_task_batch_factories=test_task_factories,
    )
    if dry.committed:
        raise RuntimeError('dry-run unexpectedly committed weights')
    _assert_equal_state(source, before)
    if set(dry.comparisons) != {'reference', 'replay_buffer'}:
        raise RuntimeError('dry-run did not produce both SAP diagnostic comparisons')
    for name, stats in dry.layer_stats.items():
        if not (-1e-5 <= stats.projection_min_eigenvalue <= stats.projection_max_eigenvalue <= 1.00001):
            raise RuntimeError(f'invalid projection eigenvalue range for {name}')
        if stats.projection_effective_rank <= 0 or not np.isfinite(stats.projection_trace):
            raise RuntimeError(f'invalid projection rank/trace for {name}')
        print(
            'LAYER_DIAGNOSTIC', name,
            f'eigen=[{stats.projection_min_eigenvalue:.8f},{stats.projection_max_eigenvalue:.8f}]',
            f'trace={stats.projection_trace:.8f}', f'rank={stats.projection_effective_rank}',
            f'norm_ratio={stats.weight_norm_ratio:.8f}',
        )
    replay = dry.comparisons['replay_buffer']
    if replay.sample_count != len(buffer_images):
        raise RuntimeError('replay diagnostic sample count mismatch')
    for group_name, group in replay.loss_tertiles.items():
        print(
            'REPLAY_TERTILE', group_name, f'count={group.sample_count}',
            f'loss_delta={group.mean_loss_delta:.8f}',
            f'accuracy_delta={group.accuracy_delta:.8f}',
            f'flip_rate={group.prediction_flip_rate:.8f}',
        )
    if len(dry.task_accuracy_comparisons) != 5:
        raise RuntimeError('expected immediate accuracy diagnostics for five CIFAR-10 tasks')
    for comparison in dry.task_accuracy_comparisons:
        if comparison.sample_count != 2000:
            raise RuntimeError(f'unexpected test task size: {comparison.sample_count}')
        print(
            'TASK_ACCURACY_DELTA', f'task={comparison.task_id}',
            f'before={comparison.accuracy_before:.8f}',
            f'after={comparison.accuracy_after:.8f}',
            f'delta={comparison.accuracy_delta:.8f}',
        )
    print(
        'DRY_RUN_OK', f'references={len(reference_images)}', f'layers={len(dry.layer_stats)}',
        f'reference_logits_delta={dry.comparisons["reference"].mean_abs_logits_delta:.8f}',
        f'replay_logits_delta={replay.mean_abs_logits_delta:.8f}',
    )

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
