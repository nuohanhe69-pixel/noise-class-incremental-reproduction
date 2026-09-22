"""Synthetic contracts for Task1 Class-4 decision-boundary analysis."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import scripts.analyze_task1_class4_decisions as class4
import scripts.compare_task1_raw_vs_centered_sap as comparison
from utils.sap import build_sap_projection_from_gram, project_linear_weight


class Task1Class4DecisionTests(unittest.TestCase):
    def test_class4_details_use_fixed_target_margin_competitor_and_gaps(self):
        labels = torch.tensor([4, 1, 4])
        pre = torch.zeros(3, 10)
        raw = torch.zeros(3, 10)
        centered = torch.zeros(3, 10)
        pre[0, [2, 3, 4]] = torch.tensor([1.0, 3.0, 5.0])
        raw[0, [2, 3, 4]] = torch.tensor([2.0, 6.0, 5.0])
        centered[0, [2, 3, 4]] = torch.tensor([1.0, 7.0, 6.0])
        pre[2, [2, 3, 4]] = torch.tensor([4.0, 1.0, 3.0])
        raw[2, [2, 3, 4]] = torch.tensor([2.0, 1.0, 4.0])
        centered[2, [2, 3, 4]] = torch.tensor([5.0, 1.0, 4.0])

        result = class4.analyze_class4_decisions(labels, pre, raw, centered)

        self.assertTrue(torch.equal(result['sample_indices'], torch.tensor([0, 2])))
        torch.testing.assert_close(result['pre_margin'], torch.tensor([2.0, -1.0]))
        self.assertTrue(torch.equal(result['pre_competitor'], torch.tensor([3, 2])))
        torch.testing.assert_close(result['pre_gap_4_vs_3'], torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(result['pre_gap_4_vs_2'], torch.tensor([4.0, -1.0]))
        self.assertEqual(result['summary']['class4_sample_count'], 2)
        self.assertEqual(result['summary']['pre_sap']['accuracy'], 50.0)
        self.assertEqual(result['summary']['raw_vs_pre']['margin_improved_count'], 1)
        self.assertEqual(result['summary']['raw_vs_pre']['margin_degraded_count'], 1)
        self.assertEqual(result['summary']['centered_vs_pre']['correct_to_wrong'], 1)
        self.assertEqual(
            result['summary']['pre_sap']['strongest_competitor_frequency'],
            {'2': 1, '3': 1},
        )

    @staticmethod
    def _write_artifacts(root: Path) -> None:
        generator = torch.Generator().manual_seed(19)
        reference = torch.randn(40, 4, generator=generator)
        reference = torch.nn.functional.normalize(reference, dim=1)
        labels = torch.arange(40) % 10
        test_features = torch.randn(40, 4, generator=generator) * 2.0
        weight_before = torch.randn(12, 4, generator=generator)
        bias = torch.randn(12, generator=generator)
        raw_gram = reference.T @ reference
        raw_projection = build_sap_projection_from_gram(raw_gram, scale=3000.0)
        weight_after = weight_before.clone()
        weight_after[:10], _ = project_linear_weight(
            weight_before[:10], raw_projection,
        )
        pre_logits = torch.nn.functional.linear(test_features, weight_before, bias)
        raw_logits = torch.nn.functional.linear(test_features, weight_after, bias)
        pre_predictions = pre_logits[:, :10].argmax(dim=1)
        raw_predictions = raw_logits[:, :10].argmax(dim=1)
        tensors = {
            'X_task1.pt': reference,
            'G_task_0.pt': raw_gram,
            'M_task_0.pt': raw_projection,
            'W_before.pt': weight_before,
            'W_after.pt': weight_after,
            'classifier_bias_before.pt': bias,
            'task1_test_features_raw.pt': test_features,
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
        pre_accuracy = float((pre_predictions == labels).float().mean() * 100)
        raw_accuracy = float((raw_predictions == labels).float().mean() * 100)
        (root / 'task1_test_decision_stats.json').write_text(json.dumps({
            'pre_accuracy_evaluator': pre_accuracy,
            'post_accuracy_evaluator': raw_accuracy,
        }))

    def _prepare_comparison(self, root: Path) -> Path:
        self._write_artifacts(root)
        comparison_output = root / 'comparison'
        comparison.run_comparison(
            root,
            comparison_output,
            decomposition_device='cpu',
            decomposition_dtype='float32',
        )
        return comparison_output / 'raw_vs_centered_summary.json'

    def test_end_to_end_reuses_centered_builder_and_writes_class4_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            comparison_summary = self._prepare_comparison(root)
            output = root / 'class4'
            with patch.object(
                class4.comparison,
                'build_centered_candidate',
                wraps=comparison.build_centered_candidate,
            ) as centered_builder:
                result_directory = class4.run_class4_analysis(
                    root,
                    output,
                    comparison_summary_path=comparison_summary,
                )

            self.assertEqual(centered_builder.call_count, 1)
            self.assertEqual(result_directory, output.resolve())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    'class4_decision_summary.json',
                    'class4_sample_details.csv',
                    'class4_pre_raw_centered_decision_analysis.png',
                },
            )
            summary = json.loads(
                (output / 'class4_decision_summary.json').read_text(),
            )
            self.assertEqual(summary['class4_sample_count'], 4)
            self.assertEqual(summary['target_class'], 4)
            self.assertEqual(summary['overall_comparison_sanity'], 'pass')
            self.assertEqual(summary['decomposition_device'], 'cpu')
            self.assertEqual(summary['decomposition_dtype'], 'float32')
            with (output / 'class4_sample_details.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(int(row['label']) == 4 for row in rows))
            self.assertEqual(
                list(rows[0]),
                class4.SAMPLE_DETAIL_COLUMNS,
            )
            self.assertGreater(
                (output / 'class4_pre_raw_centered_decision_analysis.png').stat().st_size,
                0,
            )

    def test_overall_summary_mismatch_fails_before_writing_class4_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            comparison_summary = self._prepare_comparison(root)
            payload = json.loads(comparison_summary.read_text())
            payload['centered_sap']['accuracy'] += 1.0
            comparison_summary.write_text(json.dumps(payload))
            output = root / 'class4'

            with self.assertRaisesRegex(
                AssertionError, 'existing Raw-vs-Centered summary',
            ):
                class4.run_class4_analysis(
                    root,
                    output,
                    comparison_summary_path=comparison_summary,
                )
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
