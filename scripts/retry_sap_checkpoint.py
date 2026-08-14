"""Retry only a failed SAP transaction from a saved CIFAR checkpoint.

The source checkpoint is read-only. A successful retry is written to a new
checkpoint, so reference selection and task training are never repeated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backbone.ResNetBlock import resnet18
from utils.sap_reference import deserialize_reference_memories
from utils.sap_runtime import run_sap_projection_transaction


CIFAR_NORMALIZATION = {
    'seq-cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
    'seq-cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_batches(images, *, device, mean, std, batch_size, labels=None):
    mean_tensor = torch.tensor(mean, device=device).reshape(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).reshape(1, 3, 1, 1)
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device).float()
        if images.dtype == torch.uint8:
            batch = batch.div(255)
        normalized = (batch - mean_tensor) / std_tensor
        if labels is None:
            yield normalized
        else:
            yield normalized, labels[start:start + batch_size].to(device)


def retry_sap_checkpoint(source_path: Path, output_path: Path, *, device: str | None = None) -> dict:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError('retry output must differ from the source checkpoint')
    if output_path.exists():
        raise FileExistsError(f'retry output already exists: {output_path}')
    source_digest_before = _sha256(source_path)
    saved = torch.load(source_path, map_location='cpu', weights_only=False)
    args = saved['args']
    dataset_name = args['dataset']
    if dataset_name not in CIFAR_NORMALIZATION:
        raise ValueError(f'unsupported retry dataset: {dataset_name}')
    sap_state = saved.get('sap_state')
    if not sap_state or not sap_state.get('history'):
        raise ValueError('checkpoint does not contain SAP history')
    failed_event = sap_state['history'][-1]
    if failed_event.get('status') != 'SAP_FAILED':
        raise ValueError('the latest SAP event is not a failed transaction')

    device = device or args.get('device') or ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch_device = torch.device(device)
    model = resnet18(
        num_classes=int(args['num_classes']),
        num_filters=int(args.get('num_filters', 64)),
    )
    backbone_state = {
        name.removeprefix('net.'): value
        for name, value in saved['model'].items()
        if name.startswith('net.')
    }
    model.load_state_dict(backbone_state, strict=True)
    model.to(torch_device)

    if sap_state.get('version') == 1:
        memory_state = sap_state['reference_memory']
    elif sap_state.get('version') == 2:
        memory_state = sap_state['reference_memories']
    else:
        raise ValueError('unsupported SAP state version')
    trusted_memory, _, _ = deserialize_reference_memories(memory_state)
    references = trusted_memory.items()
    if not references:
        raise ValueError('checkpoint contains no SAP references')
    reference_images = torch.stack([reference.image for reference in references])
    reference_labels = torch.tensor([reference.observed_label for reference in references])
    mean, std = CIFAR_NORMALIZATION[dataset_name]
    batch_size = int(args.get('sap_batch_size', 32))

    def reference_factory():
        return _normalized_batches(
            reference_images, device=torch_device, mean=mean, std=std,
            batch_size=batch_size,
        )

    def labeled_reference_factory():
        return _normalized_batches(
            reference_images, labels=reference_labels, device=torch_device,
            mean=mean, std=std, batch_size=batch_size,
        )

    diagnostic_factories = {'reference': labeled_reference_factory}
    buffer_state = saved.get('buffer')
    if isinstance(buffer_state, dict) and len(buffer_state.get('examples', [])):
        buffer_images = buffer_state['examples']
        buffer_labels = buffer_state['labels']

        def buffer_factory():
            return _normalized_batches(
                buffer_images, labels=buffer_labels, device=torch_device,
                mean=mean, std=std, batch_size=batch_size,
            )

        diagnostic_factories['replay_buffer'] = buffer_factory

    seen_classes = max(reference.observed_label for reference in references) + 1
    transaction = run_sap_projection_transaction(
        model,
        reference_factory,
        total_images=len(references),
        max_patches=int(args.get('sap_max_activation_patches', 20000)),
        scale=float(args.get('sap_scale', 3000.0)),
        seed=int(args.get('seed') or 0) + int(failed_event['task_id']) * 10000,
        dry_run=False,
        diagnostic_batch_factories=diagnostic_factories,
        seen_classes=seen_classes,
    )

    derived = copy.deepcopy(saved)
    projected_state = model.to('cpu').state_dict()
    for name, value in projected_state.items():
        checkpoint_name = f'net.{name}'
        if checkpoint_name not in derived['model']:
            raise KeyError(f'missing checkpoint model key: {checkpoint_name}')
        derived['model'][checkpoint_name] = value.detach().cpu().clone()
    retry_event = {
        'task_id': int(failed_event['task_id']),
        'status': 'SAP_RETRY_EXECUTED',
        'retry_source': str(source_path),
        'reference_count': len(references),
        'layer_stats': {
            name: stats.__dict__.copy() for name, stats in transaction.layer_stats.items()
        },
        'max_non_target_state_delta': transaction.max_non_target_state_delta,
        'comparisons': {
            name: {
                **comparison.__dict__,
                'loss_tertiles': {
                    group: group_stats.__dict__.copy()
                    for group, group_stats in comparison.loss_tertiles.items()
                },
            }
            for name, comparison in transaction.comparisons.items()
        },
    }
    derived['sap_state']['history'].append(retry_event)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    torch.save(derived, temporary_path)
    os.replace(temporary_path, output_path)
    if _sha256(source_path) != source_digest_before:
        raise RuntimeError('source checkpoint changed during SAP retry')
    return retry_event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    arguments = parser.parse_args()
    event = retry_sap_checkpoint(arguments.source, arguments.output, device=arguments.device)
    print(
        'SAP_RETRY_OK',
        f'references={event["reference_count"]}',
        f'layers={len(event["layer_stats"])}',
        f'max_non_target_state_delta={event["max_non_target_state_delta"]}',
    )
    for name, stats in event['layer_stats'].items():
        print(
            'SAP_RETRY_LAYER', name,
            f'patches={stats["sampled_patches"]}',
            f'dimension={stats["patch_dimension"]}',
            f'gram_device={stats["gram_device"]}',
            f'decomposition_device={stats["decomposition_device"]}',
            f'decomposition_seconds={stats["decomposition_seconds"]:.3f}',
        )


if __name__ == '__main__':
    main()
