import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from backbone.ResNetBlock import resnet18
from utils.sap import RESNET18_LATE_STAGE_CONVS
from utils.sap_reference import SAPReferenceMemory, select_task_references, split_trusted_and_pending
from utils.sap_runtime import (
    SAP_EXECUTED,
    SAPLossGroupComparison,
    SAPModelComparison,
    SAP_SKIPPED_INSUFFICIENT_REFERENCE,
    compare_models_on_labeled_batches,
    compare_models_task_accuracy,
    diagnose_reference_purity,
    extract_cifar_task_tensors,
    run_sap_projection_transaction,
    assess_sap_candidate,
    validate_reference_gate,
)


class SAPRuntimeTests(unittest.TestCase):
    @staticmethod
    def _comparison(accuracy_before, accuracy_after):
        group = SAPLossGroupComparison(
            sample_count=10,
            mean_loss_before=0.1,
            mean_loss_after=0.1,
            mean_loss_delta=0.0,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            accuracy_delta=accuracy_after - accuracy_before,
            prediction_flip_rate=0.0,
        )
        return SAPModelComparison(
            sample_count=30,
            mean_abs_logits_delta=0.1,
            max_abs_logits_delta=0.2,
            prediction_flip_rate=0.0,
            loss_tertiles={'low': group, 'mid': group, 'high': group},
        )

    def test_candidate_safety_gate_rejects_old_replay_task_regression(self):
        layer_stats = {
            'layer': SimpleNamespace(weight_norm_ratio=0.8),
        }
        decision = assess_sap_candidate(
            layer_stats,
            {
                'reference': self._comparison(1.0, 1.0),
                'replay_task_0': self._comparison(0.9, 0.6),
                'replay_task_1': self._comparison(0.8, 0.82),
            },
        )

        self.assertFalse(decision.accepted)
        self.assertIn('REPLAY_TASK_ACCURACY_DROP:replay_task_0', decision.rejection_reasons)

    def test_candidate_safety_gate_accepts_bounded_training_side_changes(self):
        decision = assess_sap_candidate(
            {'layer': SimpleNamespace(weight_norm_ratio=0.8)},
            {
                'reference': self._comparison(1.0, 0.995),
                'replay_task_0': self._comparison(0.9, 0.89),
            },
        )

        self.assertTrue(decision.accepted)

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
        trusted, _ = split_trusted_and_pending(selected)
        memory.add_task(0, trusted)
        return memory, reports

    def test_reference_gate_scales_cifar100_ratios_to_a_two_class_cifar10_task(self):
        memory, reports = self._memory_with_two_classes()

        decision = validate_reference_gate(
            memory, reports, current_class_count=2, seen_class_count=2,
        )

        self.assertEqual(decision.status, SAP_EXECUTED)
        self.assertTrue(decision.should_execute)

    def test_reference_purity_is_diagnostic_and_sample_id_aligned(self):
        memory, _ = self._memory_with_two_classes()
        references = memory.items()
        true_labels = torch.full((240,), -1)
        for reference in references:
            true_labels[reference.sample_id] = reference.observed_label
        true_labels[references[0].sample_id] = 9

        purity = diagnose_reference_purity(references, true_labels)

        self.assertAlmostEqual(purity, (len(references) - 1) / len(references))
        self.assertIsNone(diagnose_reference_purity(references, torch.zeros(1)))

    def test_pending_fallback_is_not_counted_by_the_trusted_gate(self):
        memory, reports = self._memory_with_two_classes(source='fallback')

        decision = validate_reference_gate(
            memory, reports, current_class_count=2, seen_class_count=2,
        )

        self.assertEqual(decision.status, SAP_SKIPPED_INSUFFICIENT_REFERENCE)
        self.assertFalse(decision.should_execute)
        self.assertEqual(decision.reference_count, 0)
        self.assertEqual(decision.fallback_fraction, 0.0)

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
        with torch.random.fork_rng():
            torch.manual_seed(101)
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
        self.assertEqual(transaction.max_non_target_state_delta, 0.0)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_projection_transaction_commit_changes_only_target_weights(self):
        with torch.random.fork_rng():
            torch.manual_seed(103)
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
        self.assertEqual(transaction.max_non_target_state_delta, 0.0)
        target_weights = {f'{name}.weight' for name in RESNET18_LATE_STAGE_CONVS}
        for name, value in source.state_dict().items():
            if name in target_weights:
                self.assertFalse(torch.equal(value, before[name]), name)
            else:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_projection_transaction_rolls_back_when_reference_batches_fail(self):
        with torch.random.fork_rng():
            torch.manual_seed(107)
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

    def test_model_comparison_reports_loss_tertiles_accuracy_and_prediction_flips(self):
        before = torch.nn.Linear(2, 2, bias=False)
        after = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            before.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
            after.weight.copy_(torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
        inputs = torch.tensor([
            [2.0, 0.0], [1.0, 0.0], [0.2, 0.0],
            [0.0, 0.2], [0.0, 1.0], [0.0, 2.0],
        ])
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        comparison = compare_models_on_labeled_batches(
            before,
            after,
            lambda: [(inputs, labels)],
            seen_classes=2,
        )

        self.assertEqual(comparison.sample_count, 6)
        self.assertGreater(comparison.mean_abs_logits_delta, 0)
        self.assertEqual(comparison.prediction_flip_rate, 1.0)
        self.assertEqual(set(comparison.loss_tertiles), {'low', 'mid', 'high'})
        self.assertTrue(all(group.sample_count == 2 for group in comparison.loss_tertiles.values()))
        self.assertTrue(all(group.accuracy_delta == -1.0 for group in comparison.loss_tertiles.values()))

    def test_task_accuracy_comparison_reports_immediate_delta(self):
        before = torch.nn.Linear(2, 2, bias=False)
        after = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            before.weight.copy_(torch.eye(2))
            after.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        inputs = torch.eye(2)
        labels = torch.tensor([0, 1])

        comparison = compare_models_task_accuracy(
            before, after, lambda: [(inputs, labels)], seen_classes=2, task_id=3,
        )

        self.assertEqual(comparison.task_id, 3)
        self.assertEqual(comparison.sample_count, 2)
        self.assertEqual(comparison.accuracy_before, 1.0)
        self.assertEqual(comparison.accuracy_after, 0.0)
        self.assertEqual(comparison.accuracy_delta, -1.0)


if __name__ == '__main__':
    unittest.main()
