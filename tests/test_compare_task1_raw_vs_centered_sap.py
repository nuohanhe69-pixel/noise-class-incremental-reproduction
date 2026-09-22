"""Synthetic contracts for the Task1 Raw-vs-Centered SAP comparison."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

import scripts.compare_task1_raw_vs_centered_sap as comparison
from utils.sap import build_sap_projection_from_gram, project_linear_weight


class Task1RawVsCenteredSapTests(unittest.TestCase):
    def test_centered_geometry_is_not_renormalized_and_candidate_changes_seen_rows_only(self):
        raw = torch.tensor([
            [4.0, 1.0, 0.0],
            [4.0, -1.0, 0.0],
            [4.0, 0.0, 1.0],
            [4.0, 0.0, -1.0],
        ])
        features = torch.nn.functional.normalize(raw, dim=1)
        weight = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 10

        result = comparison.build_centered_candidate(
            features,
            weight,
            class_count=10,
            scale=3000.0,
            decomposition_device='cpu',
            decomposition_dtype='float32',
        )

        expected_centered = features - features.mean(dim=0, keepdim=True)
        torch.testing.assert_close(result['centered_features'], expected_centered)
        torch.testing.assert_close(
            result['centered_gram'], expected_centered.T @ expected_centered,
        )
        self.assertFalse(torch.allclose(
            result['centered_features'].norm(dim=1),
            torch.ones(features.shape[0]),
        ))
        expected_seen, _ = project_linear_weight(
            weight[:10], result['centered_projection'],
        )
        torch.testing.assert_close(result['weight_centered'][:10], expected_seen)
        self.assertTrue(torch.equal(result['weight_centered'][10:], weight[10:]))

    def test_margin_and_paired_statistics_match_manual_values(self):
        logits_pre = torch.tensor([
            [3.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ])
        logits_candidate = torch.tensor([
            [4.0, 1.0, 0.0],
            [4.0, 3.0, 1.0],
            [3.0, 2.0, 2.5],
            [1.0, 2.0, 3.0],
        ])
        labels = torch.tensor([0, 0, 2, 1])
        pre_margin = comparison.true_class_margin(logits_pre, labels)
        candidate_margin = comparison.true_class_margin(logits_candidate, labels)

        torch.testing.assert_close(pre_margin, torch.tensor([2.0, -1.0, 1.0, -1.0]))
        stats = comparison.paired_candidate_statistics(
            labels,
            logits_pre.argmax(dim=1),
            logits_candidate.argmax(dim=1),
            candidate_margin - pre_margin,
        )
        self.assertEqual(stats['margin_improved_count'], 2)
        self.assertEqual(stats['margin_degraded_count'], 1)
        self.assertEqual(stats['prediction_changed_count'], 3)
        self.assertEqual(stats['correct_to_wrong'], 1)
        self.assertEqual(stats['wrong_to_correct'], 1)
        self.assertEqual(stats['wrong_to_wrong_prediction_changed'], 1)

    def _write_artifacts(self, root: Path) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(30, 4, generator=generator)
        features = torch.nn.functional.normalize(features, dim=1)
        labels = torch.arange(30) % 10
        weight_before = torch.randn(12, 4, generator=generator)
        bias = torch.randn(12, generator=generator)
        raw_gram = features.T @ features
        raw_projection = build_sap_projection_from_gram(raw_gram, scale=3000.0)
        weight_after = weight_before.clone()
        weight_after[:10], _ = project_linear_weight(
            weight_before[:10], raw_projection,
        )
        pre_logits = torch.nn.functional.linear(features * 2.5, weight_before, bias)
        raw_logits = torch.nn.functional.linear(features * 2.5, weight_after, bias)
        pre_predictions = pre_logits[:, :10].argmax(dim=1)
        raw_predictions = raw_logits[:, :10].argmax(dim=1)
        pre_accuracy = float((pre_predictions == labels).float().mean().item() * 100)
        raw_accuracy = float((raw_predictions == labels).float().mean().item() * 100)

        tensors = {
            'X_task1.pt': features,
            'G_task_0.pt': raw_gram,
            'M_task_0.pt': raw_projection,
            'W_before.pt': weight_before,
            'W_after.pt': weight_after,
            'classifier_bias_before.pt': bias,
            'task1_test_features_raw.pt': features * 2.5,
            'task1_test_labels.pt': labels,
            'task1_test_logits_pre_sap.pt': pre_logits,
            'task1_test_logits_post_sap.pt': raw_logits,
            'task1_test_predictions_pre_sap.pt': pre_predictions,
            'task1_test_predictions_post_sap.pt': raw_predictions,
        }
        for filename, tensor in tensors.items():
            torch.save(tensor, root / filename)
        (root / 'manifest.json').write_text(json.dumps({
            'classifier_has_bias': True,
            'test_feature_type': 'raw_classifier_input',
            'test_feature_l2_normalized': False,
        }))
        (root / 'task1_test_decision_stats.json').write_text(json.dumps({
            'pre_accuracy_evaluator': pre_accuracy,
            'post_accuracy_evaluator': raw_accuracy,
        }))
        return weight_before, bias

    def test_saved_decision_mismatch_fails_before_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_artifacts(root)
            saved_logits = torch.load(
                root / 'task1_test_logits_pre_sap.pt', weights_only=True,
            )
            saved_logits[0, 0] += 1.0
            torch.save(saved_logits, root / 'task1_test_logits_pre_sap.pt')
            output = root / 'comparison'

            with self.assertRaisesRegex(AssertionError, 'Pre-SAP logits'):
                comparison.run_comparison(root, output)
            self.assertFalse(output.exists())

    def test_end_to_end_generates_only_required_outputs_and_preserves_bias(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            weight_before, bias = self._write_artifacts(root)
            output = comparison.run_comparison(
                root,
                root / 'comparison',
                decomposition_device='cpu',
                decomposition_dtype='float32',
            )

            expected = {
                'raw_vs_centered_summary.json',
                'raw_vs_centered_per_class.csv',
                'raw_vs_centered_changed_samples.csv',
                '01_raw_vs_centered_margin_delta_distribution.png',
                '02_raw_vs_centered_sample_margin_delta.png',
                '03_raw_vs_centered_per_class_margin_delta.png',
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            summary = json.loads(
                (output / 'raw_vs_centered_summary.json').read_text(),
            )
            self.assertEqual(summary['sample_count'], 30)
            self.assertEqual(summary['seen_classes'], list(range(10)))
            self.assertEqual(summary['sap_scale'], 3000.0)
            self.assertFalse(summary['second_l2_normalization'])
            self.assertEqual(summary['decomposition_device'], 'cpu')
            self.assertEqual(summary['decomposition_dtype'], 'float32')
            self.assertEqual(summary['sanity_checks']['bias_unchanged'], 'pass')
            self.assertEqual(summary['sanity_checks']['unseen_rows_bitwise_unchanged'], 'pass')

            with (output / 'raw_vs_centered_per_class.csv').open(newline='') as handle:
                per_class = list(csv.DictReader(handle))
            self.assertEqual([int(row['class_id']) for row in per_class], list(range(10)))
            self.assertTrue(all(int(row['sample_count']) == 3 for row in per_class))
            with (output / 'raw_vs_centered_changed_samples.csv').open(newline='') as handle:
                changed = list(csv.DictReader(handle))
            self.assertTrue(all(
                row['raw_prediction_changed'] == 'True'
                or row['centered_prediction_changed'] == 'True'
                for row in changed
            ))

            # The offline comparison owns clones only; source tensors stay untouched.
            torch.testing.assert_close(
                torch.load(root / 'W_before.pt', weights_only=True), weight_before,
            )
            torch.testing.assert_close(
                torch.load(root / 'classifier_bias_before.pt', weights_only=True), bias,
            )


if __name__ == '__main__':
    unittest.main()
