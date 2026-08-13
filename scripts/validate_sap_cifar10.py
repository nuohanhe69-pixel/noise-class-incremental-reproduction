"""Validate the SAP hook and projection path on real CIFAR-10 images."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backbone.ResNetBlock import resnet18
from utils.sap import (
    RESNET18_LATE_STAGE_CONVS,
    collect_conv2d_input_gram,
    project_resnet18_from_reference_batches,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2615)
EXPECTED_STANDARD_PATCH_DIMENSIONS = {
    'layer3.0.conv1': 128 * 3 * 3,
    'layer3.0.conv2': 256 * 3 * 3,
    'layer3.0.shortcut.0': 128,
    'layer3.1.conv1': 256 * 3 * 3,
    'layer3.1.conv2': 256 * 3 * 3,
    'layer4.0.conv1': 256 * 3 * 3,
    'layer4.0.conv2': 512 * 3 * 3,
    'layer4.0.shortcut.0': 256,
    'layer4.1.conv1': 512 * 3 * 3,
    'layer4.1.conv2': 512 * 3 * 3,
}


def validate(data_root: str) -> None:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = CIFAR10(root=data_root, train=True, download=False, transform=transform)
    if len(dataset) != 50_000 or len(dataset.classes) != 10:
        raise RuntimeError('expected the complete CIFAR-10 training set')
    loader = DataLoader(
        Subset(dataset, [0, 1, 2, 3]), batch_size=2, shuffle=False, num_workers=0,
    )
    real_batches = [images for images, _ in loader]

    standard = resnet18(num_classes=10)
    for layer_index, name in enumerate(RESNET18_LATE_STAGE_CONVS):
        stats = collect_conv2d_input_gram(
            standard,
            iter(real_batches),
            name,
            total_images=4,
            max_patches=4,
            seed=100 + layer_index,
        )
        if stats.patch_dimension != EXPECTED_STANDARD_PATCH_DIMENSIONS[name]:
            raise RuntimeError(f'unexpected patch dimension for {name}: {stats.patch_dimension}')
        if stats.sampled_patches != 4 or not torch.isfinite(stats.gram).all():
            raise RuntimeError(f'invalid activation statistics for {name}')
        print(
            'HOOK_OK', name, f'dim={stats.patch_dimension}',
            f'available={stats.available_patches}', f'sampled={stats.sampled_patches}',
        )
        del stats

    source = resnet18(num_classes=10, num_filters=2)
    candidate = copy.deepcopy(source)
    source_before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    candidate_before = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    factory_calls = 0

    def batch_factory():
        nonlocal factory_calls
        factory_calls += 1
        return iter(real_batches)

    projection_stats = project_resnet18_from_reference_batches(
        source,
        candidate,
        batch_factory,
        total_images=4,
        max_patches=8,
        scale=3000.0,
        seed=0,
    )
    if factory_calls != len(RESNET18_LATE_STAGE_CONVS):
        raise RuntimeError(f'expected 10 source passes, got {factory_calls}')
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_before[name], rtol=0, atol=0)

    target_weights = {f'{name}.weight' for name in RESNET18_LATE_STAGE_CONVS}
    changed = []
    for name, value in candidate.state_dict().items():
        if name in target_weights:
            if torch.equal(value, candidate_before[name]):
                raise RuntimeError(f'target weight did not change: {name}')
            changed.append(name)
        else:
            torch.testing.assert_close(value, candidate_before[name], rtol=0, atol=0)
    if len(changed) != len(RESNET18_LATE_STAGE_CONVS):
        raise RuntimeError(f'expected 10 changed weights, got {len(changed)}')

    for name, stats in projection_stats.items():
        if stats.sampled_patches != 8 or stats.relative_weight_delta <= 0:
            raise RuntimeError(f'invalid projection statistics for {name}')
        print(
            'PROJECT_OK', name, f'dim={stats.patch_dimension}',
            f'relative_delta={stats.relative_weight_delta:.8f}',
        )
    print('REAL_CIFAR10_STAGE2_OK', f'images={len(dataset)}', f'changed_layers={len(changed)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/CIFAR10')
    validate(parser.parse_args().data_root)
