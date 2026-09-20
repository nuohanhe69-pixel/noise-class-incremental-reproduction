"""Synthetic contracts for the first-session-only SAP offline diagnostic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

import scripts.analyze_first_session_sap as diagnostic


class FirstSessionSapDiagnosticTests(unittest.TestCase):
    def test_core_analysis_uses_gram_order_and_matches_sap_identities(self):
        gram = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
        alpha = 3.0
        ratios = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64) / 10.0
        importance = alpha * ratios / ((alpha - 1.0) * ratios + 1.0)
        projection = torch.diag(importance.flip(0))
        weight_before = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [0.5, 1.5, 2.5, 3.5],
            ],
            dtype=torch.float64,
        )
        weight_after = weight_before.clone()
        weight_after[:2] = weight_before[:2] @ projection.T

        result = diagnostic.analyze_task1_geometry(
            gram=gram,
            saved_projection=projection,
            weight_before=weight_before,
            weight_after=weight_after,
            alpha=alpha,
            task1_row_count=2,
        )

        torch.testing.assert_close(
            result['eigenvalues'],
            torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(result['feature_energy_ratio'], ratios)
        torch.testing.assert_close(
            result['cumulative_energy'], ratios.cumsum(dim=0),
        )
        torch.testing.assert_close(result['sap_importance'], importance)
        torch.testing.assert_close(
            result['classifier_energy_after'],
            importance.square() * result['classifier_energy_before'],
        )
        self.assertLess(result['summary']['m_reconstruction_relative_error'], 1e-12)
        self.assertLess(result['summary']['w_projection_reconstruction_relative_error'], 1e-12)
        self.assertLess(result['summary']['theoretical_energy_relation_error'], 1e-12)
        self.assertAlmostEqual(
            result['summary']['effective_rank'],
            float(torch.exp(-(ratios * ratios.log()).sum()).item()),
        )

    def test_gram_decomposition_clamps_only_roundoff_negative_eigenvalues(self):
        almost_psd = torch.diag(
            torch.tensor([2.0, 1.0, -1e-14], dtype=torch.float64),
        )
        eigenvalues, _ = diagnostic.decompose_psd_gram(almost_psd)
        self.assertEqual(float(eigenvalues[-1].item()), 0.0)

        clearly_indefinite = torch.diag(
            torch.tensor([2.0, 1.0, -1e-3], dtype=torch.float64),
        )
        with self.assertRaisesRegex(ValueError, 'negative eigenvalue'):
            diagnostic.decompose_psd_gram(clearly_indefinite)

    def test_loader_uses_first_session_artifact_contract_and_manifest_alpha(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_dir = Path(temporary_directory)
            tensors = {
                'X_task1.pt': torch.zeros(7, 512, dtype=torch.float32),
                'G_task_0.pt': torch.eye(512, dtype=torch.float32),
                'M_task_0.pt': torch.eye(512, dtype=torch.float32),
                'W_before.pt': torch.zeros(100, 512, dtype=torch.float32),
                'W_after.pt': torch.zeros(100, 512, dtype=torch.float32),
                'trusted_labels.pt': torch.arange(7, dtype=torch.long),
                'trusted_task_ids.pt': torch.zeros(7, dtype=torch.long),
            }
            for filename, tensor in tensors.items():
                torch.save(tensor, artifact_dir / filename)
            (artifact_dir / 'manifest.json').write_text(
                json.dumps({'sap_alpha': 3000.0}), encoding='utf-8',
            )
            (artifact_dir / 'reference_stats.json').write_text(
                json.dumps({'reference_new_count': 7, 'reference_old_count': 0}),
                encoding='utf-8',
            )
            (artifact_dir / 'coverage.json').write_text(
                json.dumps({'task_counts': {'0': 7}}), encoding='utf-8',
            )
            (artifact_dir / 'accuracy.json').write_text(
                json.dumps({'pre_sap': {}, 'post_sap': {}}), encoding='utf-8',
            )

            loaded = diagnostic.load_first_session_artifacts(artifact_dir)

        self.assertEqual(loaded['alpha'], 3000.0)
        self.assertEqual(tuple(loaded['x_task1'].shape), (7, 512))
        self.assertEqual(tuple(loaded['gram'].shape), (512, 512))
        self.assertEqual(tuple(loaded['weight_before'].shape), (100, 512))
        self.assertEqual(loaded['reference_stats']['reference_old_count'], 0)

    def test_sanity_checks_reject_future_row_or_projection_mismatch(self):
        gram = torch.eye(4, dtype=torch.float64)
        projection = torch.eye(4, dtype=torch.float64) * 0.5
        weight_before = torch.ones(3, 4, dtype=torch.float64)
        valid_after = weight_before.clone()
        valid_after[:2] = weight_before[:2] @ projection.T

        future_changed = valid_after.clone()
        future_changed[2, 0] += 1.0
        with self.assertRaisesRegex(ValueError, 'rows after Task1 changed'):
            diagnostic.analyze_task1_geometry(
                gram=gram,
                saved_projection=projection,
                weight_before=weight_before,
                weight_after=future_changed,
                alpha=1.0,
                task1_row_count=2,
            )

        with self.assertRaisesRegex(ValueError, 'saved projection'):
            diagnostic.analyze_task1_geometry(
                gram=gram,
                saved_projection=projection,
                weight_before=weight_before,
                weight_after=valid_after,
                alpha=2.0,
                task1_row_count=2,
            )


if __name__ == '__main__':
    unittest.main()
