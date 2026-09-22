"""Synthetic contracts for Task1 joint SAP t-SNE visualization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import scripts.visualize_task1_sap_tsne as visualization
from utils.sap import build_sap_projection_from_gram, project_linear_weight


class _FakeTSNE:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, values):
        type(self).calls.append((self.kwargs, values.copy()))
        return values[:, :2].copy()


class Task1SapTsneTests(unittest.TestCase):
    def setUp(self):
        _FakeTSNE.calls.clear()

    def test_effective_feature_direction_matches_input_side_weight_projection(self):
        features = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        projection = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
        weight_before = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
        weight_after = weight_before @ projection.T

        effective = visualization.build_effective_features(features, projection)

        torch.testing.assert_close(effective, features @ projection)
        torch.testing.assert_close(
            effective @ weight_before.T,
            features @ weight_after.T,
        )
        self.assertFalse(torch.equal(effective, features @ projection.T))

    def test_joint_tsne_fits_once_with_fixed_config_and_preserves_block_alignment(self):
        pre = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        raw = pre + 100
        centered = pre + 200

        with patch.object(visualization, 'TSNE', _FakeTSNE):
            result = visualization.fit_joint_tsne(pre, raw, centered)

        self.assertEqual(len(_FakeTSNE.calls), 1)
        kwargs, fitted = _FakeTSNE.calls[0]
        self.assertEqual(kwargs['random_state'], 0)
        self.assertEqual(kwargs['perplexity'], 30)
        self.assertEqual(kwargs['init'], 'pca')
        self.assertEqual(kwargs['learning_rate'], 'auto')
        self.assertEqual(fitted.shape, (33, 4))
        torch.testing.assert_close(result['pre'], pre[:, :2])
        torch.testing.assert_close(result['raw'], raw[:, :2])
        torch.testing.assert_close(result['centered'], centered[:, :2])

    @staticmethod
    def _write_artifacts(root: Path) -> None:
        generator = torch.Generator().manual_seed(29)
        reference = torch.randn(40, 4, generator=generator)
        reference = torch.nn.functional.normalize(reference, dim=1)
        test_features = torch.randn(20, 4, generator=generator) * 2.0
        labels = torch.arange(20) % 10
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
        (root / 'task1_test_decision_stats.json').write_text(json.dumps({
            'pre_accuracy_evaluator': float((pre_predictions == labels).float().mean() * 100),
            'post_accuracy_evaluator': float((raw_predictions == labels).float().mean() * 100),
        }))

    def test_end_to_end_uses_all_samples_once_and_writes_png_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_artifacts(root)
            output = root / 'tsne'
            with patch.object(visualization, 'TSNE', _FakeTSNE):
                result = visualization.run_visualization(
                    root,
                    output,
                    decomposition_device='cpu',
                    decomposition_dtype='float32',
                )

            self.assertEqual(result, output.resolve())
            self.assertEqual(len(_FakeTSNE.calls), 1)
            self.assertEqual(_FakeTSNE.calls[0][1].shape, (60, 4))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    '01_task1_pre_raw_centered_joint_tsne.png',
                    'task1_tsne_summary.json',
                },
            )
            summary = json.loads((output / 'task1_tsne_summary.json').read_text())
            self.assertEqual(summary['sample_count'], 20)
            self.assertEqual(summary['joint_embedding_sample_count'], 60)
            self.assertEqual(summary['classes'], list(range(10)))
            self.assertEqual(summary['class_count'], 10)
            self.assertEqual(summary['alpha'], 3000.0)
            self.assertEqual(summary['random_state'], 0)
            self.assertEqual(summary['perplexity'], 30.0)
            self.assertEqual(summary['input_dimension'], 4)
            self.assertFalse(summary['second_l2_normalization'])
            self.assertLess(summary['raw_logits_reconstruction_error'], 1e-5)
            self.assertLess(summary['centered_logits_reconstruction_error'], 1e-5)
            self.assertGreater(
                (output / '01_task1_pre_raw_centered_joint_tsne.png').stat().st_size,
                0,
            )

    def test_raw_logit_mismatch_fails_before_tsne_or_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_artifacts(root)
            raw_logits = torch.load(
                root / 'task1_test_logits_post_sap.pt', weights_only=True,
            )
            raw_logits[0, 0] += 1.0
            torch.save(raw_logits, root / 'task1_test_logits_post_sap.pt')
            output = root / 'tsne'

            with patch.object(visualization, 'TSNE', _FakeTSNE):
                with self.assertRaisesRegex(AssertionError, 'Raw SAP logits'):
                    visualization.run_visualization(root, output)
            self.assertEqual(len(_FakeTSNE.calls), 0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
