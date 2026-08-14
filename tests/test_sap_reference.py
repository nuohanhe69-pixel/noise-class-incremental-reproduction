import unittest

import numpy as np
import torch

from utils.sap_reference import (
    GMM_FALLBACK_LOW_LOSS,
    GMM_MAIN,
    MULTISTAGE_PROMOTED,
    SAPReferenceMemory,
    SAPTrajectorySnapshot,
    assess_gmm_stability,
    build_trajectory_snapshot,
    deserialize_reference_memories,
    fit_class_robust_gmm,
    promote_pending_references,
    score_seen_class_samples,
    select_task_references,
    split_trusted_and_pending,
)


class _IndexLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', logits)

    def forward(self, sample_indices):
        return self.logits[sample_indices[:, 0, 0, 0].long()]


class SAPReferenceTests(unittest.TestCase):
    def test_stable_boundary_gmm_is_accepted_without_lowering_the_hard_floor(self):
        decision = assess_gmm_stability(
            separations=[0.94, 0.95, 0.96, 0.95, 0.94],
            component_min_weights=[0.20] * 5,
            converged_fits=5,
            attempted_fits=5,
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.acceptance_tier, 'STABLE_BOUNDARY')
        self.assertAlmostEqual(decision.separation_median, 0.95)

    def test_unstable_or_below_floor_gmm_is_rejected_with_a_reason(self):
        unstable = assess_gmm_stability(
            separations=[0.91, 0.94, 0.99, 0.92, 0.98],
            component_min_weights=[0.20] * 5,
            converged_fits=5,
            attempted_fits=5,
        )
        below_floor = assess_gmm_stability(
            separations=[0.86, 0.88, 0.89, 0.87, 0.88],
            component_min_weights=[0.20] * 5,
            converged_fits=5,
            attempted_fits=5,
        )

        self.assertFalse(unstable.accepted)
        self.assertIn('SEPARATION_UNSTABLE', unstable.rejection_reasons)
        self.assertFalse(below_floor.accepted)
        self.assertIn('SEPARATION_BELOW_FLOOR', below_floor.rejection_reasons)

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

        trusted, pending = split_trusted_and_pending(selected)
        self.assertEqual(trusted, [])
        self.assertEqual([item.sample_id for item in pending], [0, 1, 2])

    def test_v1_reference_memory_migrates_main_to_trusted_and_fallback_to_pending(self):
        count = 120
        images = torch.zeros(count, 3, 32, 32, dtype=torch.uint8)
        labels = torch.zeros(count, dtype=torch.long)
        selected, _ = select_task_references(
            images=images,
            observed_labels=labels,
            sample_ids=torch.arange(count),
            source_task_id=0,
            losses=torch.linspace(0.4, 0.401, count),
            predictions=labels,
            confidences=torch.ones(count),
            class_quota=15,
            seed=3,
        )
        legacy = SAPReferenceMemory()
        legacy.add_task(0, selected)

        trusted, pending, migrated = deserialize_reference_memories(legacy.serialize())

        self.assertTrue(migrated)
        self.assertEqual(len(trusted), 0)
        self.assertEqual(len(pending), 5)

    def test_pending_requires_consistent_multistage_evidence_before_promotion(self):
        count = 120
        labels = torch.zeros(count, dtype=torch.long)
        selected, _ = select_task_references(
            images=torch.zeros(count, 3, 32, 32, dtype=torch.uint8),
            observed_labels=labels,
            sample_ids=torch.arange(count),
            source_task_id=0,
            losses=torch.linspace(0.4, 0.401, count),
            predictions=labels,
            confidences=torch.ones(count),
            class_quota=15,
            seed=3,
        )
        _, pending = split_trusted_and_pending(selected)
        snapshots = []
        for epoch in (35, 45, 50):
            predictions = torch.zeros(len(pending), dtype=torch.long)
            if epoch == 50:
                predictions[-1] = 1
            snapshots.append(SAPTrajectorySnapshot(
                epoch=epoch,
                sample_ids=torch.tensor([item.sample_id for item in pending]),
                observed_labels=torch.zeros(len(pending), dtype=torch.long),
                losses=torch.linspace(0.10, 0.12, len(pending)) * {
                    35: 1.0, 45: 0.3, 50: 0.05,
                }[epoch],
                predictions=predictions,
                confidences=torch.full((len(pending),), 0.9),
                class_loss_quantiles=torch.full((len(pending),), 0.2),
                ogc_high_confidence=torch.ones(len(pending), dtype=torch.bool),
                abs_insertion_eligible=torch.ones(len(pending), dtype=torch.bool),
                ogc_probability_threshold=0.5,
            ))

        promoted, remaining, reports = promote_pending_references(pending, snapshots, promoted_at_task_id=0)

        self.assertEqual(len(promoted), len(pending) - 1)
        self.assertTrue(all(item.selection_source == MULTISTAGE_PROMOTED for item in promoted))
        self.assertEqual([item.sample_id for item in remaining], [pending[-1].sample_id])
        self.assertFalse(reports[-1]['promoted'])
        self.assertIn('PREDICTION_INCONSISTENT', reports[-1]['rejection_reasons'])

    def test_trajectory_snapshot_records_class_rank_ogc_and_abs_evidence(self):
        scores = type('Scores', (), {
            'losses': torch.tensor([0.1, 0.4, 0.2, 0.8]),
            'predictions': torch.tensor([0, 0, 1, 1]),
            'confidences': torch.tensor([0.9, 0.4, 0.8, 0.3]),
        })()

        snapshot = build_trajectory_snapshot(
            epoch=35,
            sample_ids=torch.arange(4),
            observed_labels=torch.tensor([0, 0, 1, 1]),
            scores=scores,
            ogc_probability_threshold=0.5,
            ogc_low_conf_weight=0.5,
            ogc_buffer_penalty_coeff=1.5,
            alpha_sample_insertion=0.5,
        )

        torch.testing.assert_close(snapshot.class_loss_quantiles, torch.tensor([0.0, 1.0, 0.0, 1.0]))
        self.assertEqual(snapshot.ogc_high_confidence.tolist(), [True, False, True, False])
        self.assertEqual(snapshot.abs_insertion_eligible.tolist(), [True, False, True, False])

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
