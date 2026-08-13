import unittest

import numpy as np
import torch

from utils.sap_reference import (
    GMM_FALLBACK_LOW_LOSS,
    GMM_MAIN,
    SAPReferenceMemory,
    fit_class_robust_gmm,
    score_seen_class_samples,
    select_task_references,
)


class _IndexLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', logits)

    def forward(self, sample_indices):
        return self.logits[sample_indices[:, 0, 0, 0].long()]


class SAPReferenceTests(unittest.TestCase):
    def test_seen_class_scoring_excludes_future_logits(self):
        logits = torch.tensor([
            [5.0, 1.0, 100.0],
            [1.0, 5.0, 100.0],
        ])
        model = _IndexLogitModel(logits)
        inputs = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1)
        labels = torch.tensor([0, 1])

        scores = score_seen_class_samples(model, [(inputs, labels)], seen_classes=2)

        expected = torch.nn.functional.cross_entropy(logits[:, :2], labels, reduction='none')
        torch.testing.assert_close(scores.losses, expected)
        self.assertEqual(scores.predictions.tolist(), [0, 1])

    def test_robust_gmm_selects_stable_low_loss_component(self):
        low = np.linspace(0.05, 0.20, 100)
        high = np.linspace(1.50, 2.00, 100)
        numpy_rng_before = np.random.get_state()
        torch_rng_before = torch.random.get_rng_state().clone()

        result = fit_class_robust_gmm(np.concatenate([low, high]), seed=7)

        self.assertGreaterEqual(result.valid_fits, 4)
        self.assertTrue(result.main_mask[:100].all())
        self.assertFalse(result.main_mask[100:].any())
        self.assertTrue((result.probability_mean[:100] >= 0.9).all())
        numpy_rng_after = np.random.get_state()
        self.assertEqual(numpy_rng_before[0], numpy_rng_after[0])
        np.testing.assert_array_equal(numpy_rng_before[1], numpy_rng_after[1])
        self.assertEqual(numpy_rng_before[2:], numpy_rng_after[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng_before)

    def test_degenerate_gmm_uses_only_prediction_agreeing_low_loss_fallback(self):
        count = 120
        losses = torch.linspace(0.4, 0.401, count)
        labels = torch.zeros(count, dtype=torch.long)
        predictions = torch.ones(count, dtype=torch.long)
        predictions[:3] = 0
        images = torch.zeros(count, 3, 32, 32, dtype=torch.uint8)

        selected, reports = select_task_references(
            images=images,
            observed_labels=labels,
            sample_ids=torch.arange(count),
            source_task_id=0,
            losses=losses,
            predictions=predictions,
            confidences=torch.full((count,), 0.5),
            class_quota=15,
            seed=3,
        )

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(item.selection_source == GMM_FALLBACK_LOW_LOSS for item in selected))
        self.assertEqual([item.sample_id for item in selected], [0, 1, 2])
        self.assertEqual(reports[0].fallback_count, 3)

    def test_selection_applies_independent_observed_class_quotas(self):
        per_class = 200
        losses = torch.tensor(
            list(np.linspace(0.05, 0.20, 100)) + list(np.linspace(1.5, 2.0, 100))
            + list(np.linspace(0.10, 0.25, 100)) + list(np.linspace(1.6, 2.1, 100)),
        )
        labels = torch.tensor([0] * per_class + [1] * per_class)
        images = torch.zeros(2 * per_class, 3, 32, 32, dtype=torch.uint8)

        selected, reports = select_task_references(
            images=images,
            observed_labels=labels,
            sample_ids=torch.arange(2 * per_class),
            source_task_id=0,
            losses=losses,
            predictions=labels.clone(),
            confidences=torch.full((2 * per_class,), 0.9),
            class_quota=15,
            seed=5,
        )

        self.assertEqual([sum(item.observed_label == label for item in selected) for label in (0, 1)], [15, 15])
        self.assertTrue(all(item.selection_source == GMM_MAIN for item in selected))
        self.assertEqual({report.observed_label for report in reports}, {0, 1})

    def test_reference_memory_preserves_old_tasks_and_round_trips(self):
        memory = SAPReferenceMemory()
        images = torch.zeros(120, 3, 32, 32, dtype=torch.uint8)
        labels = torch.zeros(120, dtype=torch.long)
        selected, _ = select_task_references(
            images=images,
            observed_labels=labels,
            sample_ids=torch.arange(120),
            source_task_id=0,
            losses=torch.cat([torch.linspace(0.05, 0.2, 60), torch.linspace(1.5, 2.0, 60)]),
            predictions=labels,
            confidences=torch.ones(120),
            class_quota=5,
            seed=9,
        )
        memory.add_task(0, selected)
        old_state = memory.serialize()

        new_selected = [item.clone_for_task(1, sample_id=item.sample_id + 1000) for item in selected]
        memory.add_task(1, new_selected)

        current_old_task = memory.serialize()['tasks']['0']
        for current, original in zip(current_old_task, old_state['tasks']['0']):
            torch.testing.assert_close(current['image'], original['image'], rtol=0, atol=0)
            self.assertEqual(
                {key: value for key, value in current.items() if key != 'image'},
                {key: value for key, value in original.items() if key != 'image'},
            )
        restored = SAPReferenceMemory.deserialize(memory.serialize())
        self.assertEqual(len(restored), len(memory))
        for restored_item, original_item in zip(restored.items(), memory.items()):
            torch.testing.assert_close(restored_item.image, original_item.image, rtol=0, atol=0)
            self.assertEqual(restored_item.observed_label, original_item.observed_label)
            self.assertEqual(restored_item.source_task_id, original_item.source_task_id)
            self.assertEqual(restored_item.sample_id, original_item.sample_id)
            self.assertEqual(restored_item.selection_source, original_item.selection_source)
        self.assertTrue(all(item.image.dtype == torch.uint8 and item.image.device.type == 'cpu' for item in restored.items()))


if __name__ == '__main__':
    unittest.main()
