import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from backbone.ResNetBlock import resnet18
from utils.sap import RESNET18_LATE_STAGE_CONVS
from utils.sap_reference import SAPReferenceMemory, select_task_references
from utils.sap_runtime import (
    SAP_EXECUTED,
    SAP_SKIPPED_EXCESSIVE_FALLBACK,
    extract_cifar_task_tensors,
    run_sap_projection_transaction,
    validate_reference_gate,
)


class SAPRuntimeTests(unittest.TestCase):
    def _memory_with_two_classes(self, source='GMM_MAIN'):
        memory = SAPReferenceMemory()
        count = 240
        labels = torch.tensor([0] * 120 + [1] * 120)
        losses = torch.cat([
            torch.linspace(0.05, 0.20, 60), torch.linspace(1.5, 2.0, 60),
            torch.linspace(0.05, 0.20, 60), torch.linspace(1.5, 2.0, 60),
        ])
        predictions = labels.clone()
        if source != 'GMM_MAIN':
            losses.fill_(0.4)
            predictions.fill_(3)
            predictions[:5] = 0
            predictions[120:125] = 1
        selected, reports = select_task_references(
            images=torch.zeros(count, 3, 32, 32, dtype=torch.uint8),
            observed_labels=labels,
            sample_ids=torch.arange(count),
            source_task_id=0,
            losses=losses,
            predictions=predictions,
            confidences=torch.ones(count),
            class_quota=15,
            seed=0,
        )
        memory.add_task(0, selected)
        return memory, reports

    def test_reference_gate_scales_cifar100_ratios_to_a_two_class_cifar10_task(self):
        memory, reports = self._memory_with_two_classes()

        decision = validate_reference_gate(
            memory, reports, current_class_count=2, seen_class_count=2,
        )

        self.assertEqual(decision.status, SAP_EXECUTED)
        self.assertTrue(decision.should_execute)

    def test_reference_gate_rejects_excessive_fallback(self):
        memory, reports = self._memory_with_two_classes(source='fallback')

        decision = validate_reference_gate(
            memory, reports, current_class_count=2, seen_class_count=2,
        )

        self.assertEqual(decision.status, SAP_SKIPPED_EXCESSIVE_FALLBACK)
        self.assertFalse(decision.should_execute)

    def test_extract_cifar_task_tensors_preserves_observed_labels_and_sample_ids(self):
        wrapped_dataset = SimpleNamespace(
            data=np.arange(2 * 32 * 32 * 3, dtype=np.uint8).reshape(2, 32, 32, 3),
            targets=torch.tensor([8, 9]),
            indexes=np.array([101, 202]),
        )

        images, labels, sample_ids = extract_cifar_task_tensors(wrapped_dataset)

        self.assertEqual(images.shape, (2, 3, 32, 32))
        self.assertEqual(images.dtype, torch.uint8)
        self.assertEqual(labels.tolist(), [8, 9])
        self.assertEqual(sample_ids.tolist(), [101, 202])

    def test_projection_transaction_dry_run_preserves_every_source_parameter(self):
        source = resnet18(num_classes=10, num_filters=1)
        before = {name: value.detach().clone() for name, value in source.state_dict().items()}
        inputs = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(31))

        transaction = run_sap_projection_transaction(
            source,
            lambda: [inputs],
            total_images=2,
            max_patches=4,
            scale=10.0,
            seed=0,
            dry_run=True,
        )

        self.assertFalse(transaction.committed)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_projection_transaction_commit_changes_only_target_weights(self):
        source = resnet18(num_classes=10, num_filters=1)
        before = {name: value.detach().clone() for name, value in source.state_dict().items()}
        inputs = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(37))

        transaction = run_sap_projection_transaction(
            source,
            lambda: [inputs],
            total_images=2,
            max_patches=4,
            scale=10.0,
            seed=0,
            dry_run=False,
        )

        self.assertTrue(transaction.committed)
        target_weights = {f'{name}.weight' for name in RESNET18_LATE_STAGE_CONVS}
        for name, value in source.state_dict().items():
            if name in target_weights:
                self.assertFalse(torch.equal(value, before[name]), name)
            else:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_projection_transaction_rolls_back_when_reference_batches_fail(self):
        source = resnet18(num_classes=10, num_filters=1)
        before = {name: value.detach().clone() for name, value in source.state_dict().items()}

        with self.assertRaisesRegex(ValueError, 'contain 0 images'):
            run_sap_projection_transaction(
                source,
                lambda: [],
                total_images=2,
                max_patches=4,
                scale=10.0,
                seed=0,
                dry_run=False,
            )

        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
