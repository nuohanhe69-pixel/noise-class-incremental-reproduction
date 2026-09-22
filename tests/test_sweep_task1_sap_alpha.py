"""Synthetic contracts for the offline Task1 SAP alpha sweep."""

from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import scripts.sweep_task1_sap_alpha as sweep
from utils.sap import build_sap_projection_from_gram, project_linear_weight


class Task1SapAlphaSweepTests(unittest.TestCase):
    def test_fixed_alpha_grid_and_raw_centered_candidates_preserve_scope(self):
        self.assertEqual(
            sweep.ALPHA_GRID,
            (1, 10, 30, 100, 300, 1000, 3000, 10000),
        )
        raw = torch.tensor([
            [4.0, 1.0, 0.0], [4.0, -1.0, 0.0],
            [4.0, 0.0, 1.0], [4.0, 0.0, -1.0],
        ])
        reference = torch.nn.functional.normalize(raw, dim=1)
        raw_gram, centered_gram = sweep.prepare_raw_and_centered_grams(
            reference,
            decomposition_device='cpu',
            decomposition_dtype='float32',
        )
        weight = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 10
        bias = torch.arange(12, dtype=torch.float32)
        bias_before = bias.clone()

        raw_candidate = sweep.build_candidate(
            weight, raw_gram, alpha=100, class_count=10,
        )
        centered_candidate = sweep.build_candidate(
            weight, centered_gram, alpha=100, class_count=10,
        )

        torch.testing.assert_close(raw_gram, reference.T @ reference)
        expected_centered = reference - reference.mean(dim=0, keepdim=True)
        torch.testing.assert_close(centered_gram, expected_centered.T @ expected_centered)
        self.assertFalse(torch.allclose(
            expected_centered.norm(dim=1), torch.ones(reference.shape[0]),
        ))
        self.assertTrue(torch.equal(raw_candidate['weight'][10:], weight[10:]))
        self.assertTrue(torch.equal(centered_candidate['weight'][10:], weight[10:]))
        self.assertTrue(torch.equal(bias, bias_before))

    @staticmethod
    def _write_artifacts(root: Path) -> None:
        generator = torch.Generator().manual_seed(37)
        reference = torch.randn(40, 4, generator=generator)
        reference = torch.nn.functional.normalize(reference, dim=1)
        features = torch.randn(40, 4, generator=generator) * 2.0
        labels = torch.arange(40) % 10
        weight_before = torch.randn(12, 4, generator=generator)
        bias = torch.randn(12, generator=generator)
        raw_gram = reference.T @ reference
        raw_projection = build_sap_projection_from_gram(raw_gram, scale=3000.0)
        weight_after = weight_before.clone()
        weight_after[:10], _ = project_linear_weight(
            weight_before[:10], raw_projection,
        )
        pre_logits = torch.nn.functional.linear(features, weight_before, bias)
        raw_logits = torch.nn.functional.linear(features, weight_after, bias)
        pre_predictions = pre_logits[:, :10].argmax(dim=1)
        raw_predictions = raw_logits[:, :10].argmax(dim=1)
        tensors = {
            'X_task1.pt': reference,
            'G_task_0.pt': raw_gram,
            'M_task_0.pt': raw_projection,
            'W_before.pt': weight_before,
            'W_after.pt': weight_after,
            'classifier_bias_before.pt': bias,
            'task1_test_features_raw.pt': features,
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
            'pre_accuracy_evaluator': float((pre_predictions == labels).float().mean() * 100),
            'post_accuracy_evaluator': float((raw_predictions == labels).float().mean() * 100),
        }))

    def test_end_to_end_runs_both_paths_and_writes_required_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_artifacts(root)
            output = root / 'sweep'
            with patch.object(
                sweep,
                'build_sap_projection_from_gram',
                wraps=build_sap_projection_from_gram,
            ) as projection_builder:
                result = sweep.run_sweep(
                    root,
                    output,
                    decomposition_device='cpu',
                    decomposition_dtype='float32',
                )

            self.assertEqual(result, output.resolve())
            self.assertEqual(projection_builder.call_count, len(sweep.ALPHA_GRID) * 2)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    'task1_sap_alpha_sweep_summary.csv',
                    'task1_sap_alpha_sweep_per_class.csv',
                    'task1_sap_alpha_sweep_summary.json',
                    '01_alpha_vs_accuracy.png',
                    '02_alpha_vs_mean_margin.png',
                },
            )
            payload = json.loads(
                (output / 'task1_sap_alpha_sweep_summary.json').read_text(),
            )
            self.assertEqual(payload['alpha_grid'], list(sweep.ALPHA_GRID))
            self.assertEqual(len(payload['results']), len(sweep.ALPHA_GRID) * 2)
            self.assertEqual(
                {row['variant'] for row in payload['results']},
                {'raw', 'centered'},
            )
            self.assertTrue(payload['sanity_checks']['alpha_3000_raw_reconstruction'])
            self.assertTrue(payload['sanity_checks']['bias_unchanged'])
            self.assertTrue(payload['sanity_checks']['unseen_rows_unchanged'])
            self.assertFalse(payload['training_performed'])
            self.assertFalse(payload['second_l2_normalization'])

            with (output / 'task1_sap_alpha_sweep_summary.csv').open(newline='') as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 1 + len(sweep.ALPHA_GRID) * 2)
            self.assertEqual(summary_rows[0]['variant'], 'pre_sap')
            self.assertEqual(summary_rows[0]['alpha'], '')
            self.assertTrue({
                'variant', 'alpha', 'accuracy', 'mean_margin', 'median_margin',
                'prediction_changed_count', 'correct_to_wrong', 'wrong_to_correct',
                'wrong_to_wrong_prediction_changed',
            }.issubset(summary_rows[0]))
            with (output / 'task1_sap_alpha_sweep_per_class.csv').open(newline='') as handle:
                class_rows = list(csv.DictReader(handle))
            self.assertEqual(
                len(class_rows),
                10 + len(sweep.ALPHA_GRID) * 2 * 10,
            )
            self.assertEqual(class_rows[0]['variant'], 'pre_sap')
            self.assertEqual(class_rows[0]['alpha'], '')
            self.assertTrue({
                'variant', 'alpha', 'class_id', 'sample_count', 'accuracy',
                'accuracy_delta_vs_pre', 'mean_margin',
                'mean_margin_delta_vs_pre',
            }.issubset(class_rows[0]))

            source = inspect.getsource(sweep)
            self.assertNotIn('initialize(', source)
            self.assertNotIn('train(', source)

    def test_alpha_3000_raw_mismatch_fails_before_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_artifacts(root)
            saved_projection = torch.load(root / 'M_task_0.pt', weights_only=True)
            saved_projection[0, 0] += 0.1
            torch.save(saved_projection, root / 'M_task_0.pt')
            output = root / 'sweep'

            with self.assertRaisesRegex(AssertionError, 'alpha=3000 Raw'):
                sweep.run_sweep(
                    root,
                    output,
                    decomposition_device='cpu',
                    decomposition_dtype='float32',
                )
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
