"""Synthetic contracts for the first-session-only SAP offline diagnostic."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import scripts.analyze_first_session_sap as diagnostic


class FirstSessionSapDiagnosticTests(unittest.TestCase):
    def test_float32_tail_error_uses_numerical_floor_and_is_separation_inactive(self):
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        tail = 1e-4
        features = torch.tensor(
            [
                [1.0, 1.0, tail],
                [1.0, -1.0, -tail],
                [-1.0, 1.0, tail],
                [-1.0, -1.0, -tail],
            ],
            dtype=torch.float32,
        )
        raw_gram = features.T @ features
        projection_energy = raw_gram.diag()
        raw_eigenvalues = projection_energy.clone()
        raw_eigenvalues[-1] *= 1.02

        result = diagnostic.analyze_raw_direction_class_structure(
            x_task1=features,
            trusted_labels=labels,
            raw_gram=raw_gram,
            raw_eigenvalues=raw_eigenvalues,
            raw_eigenvectors=torch.eye(3, dtype=torch.float32),
            decomposition_device='cpu',
            decomposition_dtype='float32',
            sanity_tolerance=1e-4,
        )

        self.assertLess(
            result['summary']['raw_energy_decomposition_relative_error'], 1e-4,
        )
        self.assertGreater(
            result['summary']['raw_energy_decomposition_max_direction_error'],
            1e-2,
        )
        self.assertFalse(result['class_separation_active'][-1])
        self.assertTrue(torch.isnan(result['class_separation_ratio'][-1]))
        self.assertTrue(torch.isnan(result['common_mean_fraction'][-1]))

    def test_float32_mixed_tolerance_still_rejects_material_direction_error(self):
        dimension = 101
        features = torch.eye(dimension, dtype=torch.float32)
        labels = torch.cat((
            torch.zeros(50, dtype=torch.long),
            torch.ones(dimension - 50, dtype=torch.long),
        ))
        raw_gram = features.T @ features
        raw_eigenvalues = torch.ones(dimension, dtype=torch.float32)
        raw_eigenvalues[0] = 1.001

        with self.assertRaisesRegex(
            ValueError, 'Raw directional energy decomposition failed',
        ):
            diagnostic.analyze_raw_direction_class_structure(
                x_task1=features,
                trusted_labels=labels,
                raw_gram=raw_gram,
                raw_eigenvalues=raw_eigenvalues,
                raw_eigenvectors=torch.eye(dimension, dtype=torch.float32),
                decomposition_device='cpu',
                decomposition_dtype='float32',
                sanity_tolerance=1e-4,
            )

    def test_raw_direction_class_energy_decomposition_and_activity(self):
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        within_sign = torch.tensor(
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
        )
        features = torch.stack((
            torch.full((8,), 3.0, dtype=torch.float64),
            torch.where(labels == 0, 2.0, -2.0).to(torch.float64),
            within_sign,
            torch.zeros(8, dtype=torch.float64),
        ), dim=1)
        features = torch.nn.functional.normalize(features, dim=1)
        raw_gram = features.T @ features
        raw_eigenvalues = raw_gram.diag()
        raw_eigenvectors = torch.eye(4, dtype=torch.float64)

        result = diagnostic.analyze_raw_direction_class_structure(
            x_task1=features,
            trusted_labels=labels,
            raw_gram=raw_gram,
            raw_eigenvalues=raw_eigenvalues,
            raw_eigenvectors=raw_eigenvectors,
            decomposition_device='cpu',
            decomposition_dtype='float64',
            sanity_tolerance=1e-10,
        )

        torch.testing.assert_close(
            result['sample_projections'].square().sum(dim=0), raw_eigenvalues,
        )
        torch.testing.assert_close(
            result['common_mean_energy']
            + result['between_class_energy']
            + result['within_class_energy'],
            raw_eigenvalues,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertGreater(result['common_mean_energy'][0], 5.0)
        self.assertLess(result['between_class_energy'][0], 1e-12)
        self.assertGreater(result['class_separation_ratio'][1], 0.999)
        self.assertLess(result['class_separation_ratio'][2], 1e-12)
        self.assertFalse(result['class_separation_active'][3])
        self.assertTrue(torch.isnan(result['class_separation_ratio'][3]))
        self.assertLess(
            result['summary']['projection_energy_reconstruction_relative_error'],
            1e-10,
        )
        self.assertLess(
            result['summary']['raw_energy_decomposition_relative_error'],
            1e-10,
        )
        active_raw_energy = raw_eigenvalues > 0
        torch.testing.assert_close(
            result['common_mean_fraction'][active_raw_energy]
            + result['between_class_fraction'][active_raw_energy]
            + result['within_class_fraction'][active_raw_energy],
            torch.ones(int(active_raw_energy.sum()), dtype=torch.float64),
        )
        self.assertTrue({
            'common_mean_energy',
            'between_class_energy',
            'within_class_energy',
            'residual_variation_energy',
            'common_mean_fraction',
            'between_class_fraction',
            'within_class_fraction',
            'class_separation_active',
            'class_separation_ratio',
            'between_class_energy_share',
        }.issubset(diagnostic.DIRECTION_COLUMNS))
        self.assertTrue({
            'common_mean_energy_total',
            'between_class_energy_total',
            'within_class_energy_total',
            'common_mean_energy_fraction_total',
            'between_class_energy_fraction_total',
            'within_class_energy_fraction_total',
            'global_class_separation_ratio',
            'raw_energy_decomposition_relative_error',
            'raw_energy_decomposition_max_direction_error',
            'projection_energy_reconstruction_relative_error',
        }.issubset(result['summary']))

    def test_feature_centering_removes_strong_common_mean_without_renormalizing(self):
        variations = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float64,
        )
        features = torch.cat(
            (torch.full((len(variations), 1), 3.0, dtype=torch.float64), variations),
            dim=1,
        )
        features = torch.nn.functional.normalize(features, dim=1)
        raw_gram = features.T @ features
        raw_eigenvalues, raw_eigenvectors = diagnostic.decompose_psd_gram(
            raw_gram,
            decomposition_device='cpu',
            decomposition_dtype='float64',
        )
        raw_ratios = raw_eigenvalues / raw_eigenvalues.sum()
        positive_raw_ratios = raw_ratios[raw_ratios > 0]
        raw_effective_rank = torch.exp(
            -(positive_raw_ratios * positive_raw_ratios.log()).sum(),
        )

        centered = diagnostic.analyze_feature_centering(
            x_task1=features,
            raw_gram=raw_gram,
            raw_eigenvectors=raw_eigenvectors,
            decomposition_device='cpu',
            decomposition_dtype='float64',
            sanity_tolerance=1e-10,
        )

        self.assertGreater(float(raw_ratios[0].item()), 0.85)
        self.assertGreater(centered['summary']['raw_top1_mean_cosine_squared'], 0.999)
        self.assertLess(
            centered['summary']['centered_top1_feature_energy_ratio'],
            float(raw_ratios[0].item()),
        )
        self.assertGreater(
            centered['summary']['centered_effective_rank'],
            float(raw_effective_rank.item()),
        )
        torch.testing.assert_close(
            centered['centered_gram'],
            raw_gram - len(features) * torch.outer(features.mean(dim=0), features.mean(dim=0)),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertLess(
            centered['summary']['centered_gram_identity_relative_error'], 1e-10,
        )
        self.assertLess(
            centered['summary']['trace_decomposition_absolute_error'], 1e-10,
        )
        expected_centered = features - features.mean(dim=0, keepdim=True)
        torch.testing.assert_close(centered['centered_features'], expected_centered)
        self.assertFalse(torch.allclose(
            centered['centered_features'].norm(dim=1),
            torch.ones(len(features), dtype=torch.float64),
        ))

    def test_direction_summary_maps_eigenvalue_header_to_eigenvalues_result(self):
        result = {
            column: torch.tensor([1.0])
            for column in diagnostic.DIRECTION_COLUMNS
            if column not in ('direction_rank', 'eigenvalue')
        }
        result['eigenvalues'] = torch.tensor([3.5])

        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = diagnostic._write_direction_summary(
                result, Path(temporary_directory),
            )
            with csv_path.open(newline='', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, list(diagnostic.DIRECTION_COLUMNS))
        self.assertNotIn('eigenvalues', reader.fieldnames)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['direction_rank'], '1')
        self.assertEqual(float(rows[0]['eigenvalue']), 3.5)

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
            decomposition_device='cpu',
            decomposition_dtype='float64',
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
        torch.testing.assert_close(
            result['centered_classifier_energy_after'],
            importance.square() * result['centered_classifier_energy_before'],
        )
        self.assertLess(result['summary']['m_reconstruction_relative_error'], 1e-12)
        self.assertLess(result['summary']['w_projection_reconstruction_relative_error'], 1e-12)
        self.assertLess(result['summary']['theoretical_energy_relation_error'], 1e-12)
        self.assertLess(
            result['summary']['centered_theoretical_energy_relation_error'], 1e-12,
        )
        self.assertEqual(result['summary']['decomposition_device'], 'cpu')
        self.assertEqual(result['summary']['decomposition_dtype'], 'float64')
        self.assertTrue({
            'centered_classifier_energy_before',
            'centered_classifier_energy_after',
            'centered_classifier_energy_ratio',
            'centered_classifier_energy_loss',
            'centered_classifier_energy_loss_ratio',
            'classifier_energy_active',
            'centered_classifier_energy_active',
        }.issubset(diagnostic.DIRECTION_COLUMNS))
        self.assertTrue({
            'centered_classifier_energy_before_total',
            'centered_classifier_energy_after_total',
            'centered_classifier_energy_total_ratio',
            'centered_theoretical_energy_relation_error',
        }.issubset(result['summary']))
        self.assertAlmostEqual(
            result['summary']['effective_rank'],
            float(torch.exp(-(ratios * ratios.log()).sum()).item()),
        )

    def test_common_classifier_component_changes_absolute_but_not_centered_energy(self):
        gram = torch.diag(
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        )
        alpha = 3.0
        ratios = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64) / 10.0
        importance = alpha * ratios / ((alpha - 1.0) * ratios + 1.0)
        projection = torch.diag(importance.flip(0))
        weight_before = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [2.0, 1.0, 4.0, 3.0],
            ],
            dtype=torch.float64,
        )
        weight_after = weight_before @ projection.T
        common_component = torch.tensor(
            [[10.0, -7.0, 5.0, 3.0]], dtype=torch.float64,
        )
        shifted_before = weight_before + common_component
        shifted_after = shifted_before @ projection.T

        original = diagnostic.analyze_task1_geometry(
            gram=gram,
            saved_projection=projection,
            weight_before=weight_before,
            weight_after=weight_after,
            alpha=alpha,
            task1_row_count=3,
        )
        shifted = diagnostic.analyze_task1_geometry(
            gram=gram,
            saved_projection=projection,
            weight_before=shifted_before,
            weight_after=shifted_after,
            alpha=alpha,
            task1_row_count=3,
        )

        self.assertFalse(torch.allclose(
            original['classifier_energy_before'],
            shifted['classifier_energy_before'],
        ))
        torch.testing.assert_close(
            original['centered_classifier_energy_before'],
            shifted['centered_classifier_energy_before'],
        )
        torch.testing.assert_close(
            original['centered_classifier_energy_after'],
            shifted['centered_classifier_energy_after'],
        )

    def test_zero_before_energy_is_inactive_not_reported_as_full_loss(self):
        gram = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
        alpha = 3.0
        ratios = torch.tensor([2.0, 1.0], dtype=torch.float64) / 3.0
        importance = alpha * ratios / ((alpha - 1.0) * ratios + 1.0)
        projection = torch.diag(importance)
        weight_before = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64,
        )
        weight_after = weight_before @ projection.T

        result = diagnostic.analyze_task1_geometry(
            gram=gram,
            saved_projection=projection,
            weight_before=weight_before,
            weight_after=weight_after,
            alpha=alpha,
            task1_row_count=2,
        )

        inactive_direction = 1
        self.assertFalse(result['classifier_energy_active'][inactive_direction])
        self.assertTrue(torch.isnan(
            result['classifier_energy_ratio'][inactive_direction],
        ))
        self.assertTrue(torch.isnan(
            result['classifier_energy_loss_ratio'][inactive_direction],
        ))
        self.assertFalse(
            result['centered_classifier_energy_active'][inactive_direction],
        )
        self.assertTrue(torch.isnan(
            result['centered_classifier_energy_ratio'][inactive_direction],
        ))
        self.assertTrue(torch.isnan(
            result['centered_classifier_energy_loss_ratio'][inactive_direction],
        ))

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

    def test_decomposition_dtype_is_explicit_and_cuda_never_silently_falls_back(self):
        gram = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))

        artifact_dtype_values, _ = diagnostic.decompose_psd_gram(
            gram,
            decomposition_device='cpu',
            decomposition_dtype='artifact',
        )
        float32_values, _ = diagnostic.decompose_psd_gram(
            gram,
            decomposition_device='cpu',
            decomposition_dtype='float32',
        )

        self.assertEqual(artifact_dtype_values.device.type, 'cpu')
        self.assertEqual(artifact_dtype_values.dtype, torch.float64)
        self.assertEqual(float32_values.device.type, 'cpu')
        self.assertEqual(float32_values.dtype, torch.float32)
        with patch('scripts.analyze_first_session_sap.torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA.*unavailable'):
                diagnostic.decompose_psd_gram(
                    gram,
                    decomposition_device='cuda',
                    decomposition_dtype='float32',
                )

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
