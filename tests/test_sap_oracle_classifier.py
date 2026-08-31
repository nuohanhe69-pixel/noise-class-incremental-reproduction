"""Unit tests for the oracle task-boundary SAP classifier primitives."""

import unittest

import torch
from torch import nn

from utils.sap import (
    collect_class_balanced_classifier_input_gram,
    project_linear_weight,
)


class _TinyClassifierNet(nn.Module):
    """A minimal net whose forward produces a 4-d feature fed to an nn.Linear."""

    def __init__(self, in_features: int = 4, out_features: int = 3):
        super().__init__()
        self.feature_dim = in_features
        self.classifier = nn.Linear(in_features, out_features, bias=True)

    def forward(self, inputs):
        # Mimic ResNet's global avg pool + flatten: in tests we feed already-flat
        # tensors; here just pass through to the classifier.
        return self.classifier(inputs)


class CollectClassBalancedClassifierInputGramTests(unittest.TestCase):
    def test_matches_explicit_class_balanced_gram_with_unequal_class_counts(self):
        model = _TinyClassifierNet(in_features=3, out_features=2)
        class_zero = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
        class_one = torch.tensor([
            [4.0, 0.0, 1.0], [3.0, 1.0, 0.0], [2.0, 2.0, 1.0],
            [1.0, 3.0, 2.0], [0.0, 4.0, 3.0],
        ])
        features = torch.cat([class_zero, class_one])
        labels = torch.tensor([0, 0, 1, 1, 1, 1, 1])

        stats = collect_class_balanced_classifier_input_gram(
            model, [(features[:3], labels[:3]), (features[3:], labels[3:])],
            total_images=7, seen_classes=2,
        )

        expected = 0.5 * (
            class_zero.T @ class_zero / len(class_zero)
            + class_one.T @ class_one / len(class_one)
        )
        ordinary = features.T @ features
        torch.testing.assert_close(stats.gram, expected)
        self.assertFalse(torch.allclose(stats.gram, ordinary))
        self.assertEqual(stats.per_class_counts, (2, 5))
        self.assertEqual(sum(stats.per_class_counts), len(features))
        self.assertEqual(stats.gram.shape, (3, 3))

    def test_all_samples_across_batches_contribute_without_sampling(self):
        model = _TinyClassifierNet(in_features=2, out_features=2)
        features = torch.tensor([
            [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
            [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0],
        ])
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 1])

        stats = collect_class_balanced_classifier_input_gram(
            model, [(features[:2], labels[:2]), (features[2:5], labels[2:5]),
                    (features[5:], labels[5:])],
            total_images=7, seen_classes=2,
        )

        expected = 0.5 * (
            features[:3].T @ features[:3] / 3
            + features[3:].T @ features[3:] / 4
        )
        torch.testing.assert_close(stats.gram, expected)
        self.assertEqual(stats.per_class_counts, (3, 4))

    def test_total_images_mismatch_raises(self):
        model = _TinyClassifierNet(in_features=3, out_features=2)
        features = torch.randn(4, 3)
        with self.assertRaises(ValueError):
            collect_class_balanced_classifier_input_gram(
                model, [(features, torch.tensor([0, 0, 1, 1]))],
                total_images=5, seen_classes=2,
            )

    def test_missing_seen_class_fails_closed(self):
        model = _TinyClassifierNet(in_features=3, out_features=3)
        features = torch.randn(4, 3)
        with self.assertRaisesRegex(ValueError, r'missing.*\[2\]'):
            collect_class_balanced_classifier_input_gram(
                model, [(features, torch.tensor([0, 0, 1, 1]))],
                total_images=4, seen_classes=3,
            )

    def test_missing_classifier_module_raises(self):
        class _Headless(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, kernel_size=1)
            def forward(self, x):
                return self.conv(x)

        with self.assertRaises(ValueError):
            collect_class_balanced_classifier_input_gram(
                _Headless(), [(torch.randn(2, 1, 4, 4), torch.tensor([0, 1]))],
                total_images=2, seen_classes=2,
            )

    def test_restores_training_state_after_collection(self):
        model = _TinyClassifierNet(in_features=3, out_features=2)
        model.train()  # put some modules in train mode
        self.assertTrue(model.training)
        self.assertTrue(model.classifier.training)

        collect_class_balanced_classifier_input_gram(
            model, [(torch.randn(2, 3), torch.tensor([0, 1]))],
            total_images=2, seen_classes=2,
        )

        # Training state must be restored after the hook runs.
        self.assertTrue(model.training)
        self.assertTrue(model.classifier.training)


class ProjectLinearWeightTests(unittest.TestCase):
    def test_projection_applies_W_M_T_and_returns_stats(self):
        generator = torch.Generator().manual_seed(11)
        out_features, in_features = 3, 5
        weight = torch.randn(out_features, in_features, generator=generator)
        projection = torch.eye(in_features) + 0.01 * torch.randn(in_features, in_features,
                                                                  generator=generator)

        projected, stats = project_linear_weight(
            weight, projection, n_seen_classes=2,
        )

        torch.testing.assert_close(projected[:2], weight[:2] @ projection.T)
        torch.testing.assert_close(projected[2:], weight[2:], rtol=0, atol=0)
        self.assertIn('relative_weight_delta', stats)
        self.assertIn('weight_norm_ratio', stats)
        self.assertGreater(stats['relative_weight_delta'], 0.0)
        self.assertTrue(torch.isfinite(projected).all())

    def test_identity_projection_leaves_weight_unchanged(self):
        weight = torch.randn(4, 6)
        projection = torch.eye(6)

        projected, stats = project_linear_weight(
            weight, projection, n_seen_classes=3,
        )

        torch.testing.assert_close(projected, weight)
        self.assertAlmostEqual(stats['relative_weight_delta'], 0.0, places=7)
        self.assertAlmostEqual(stats['weight_norm_ratio'], 1.0, places=6)

    def test_wrong_projection_shape_raises(self):
        weight = torch.randn(2, 4)
        with self.assertRaises(ValueError):
            project_linear_weight(weight, torch.eye(5), n_seen_classes=2)

    def test_non_finite_projection_raises(self):
        weight = torch.randn(2, 3)
        bad = torch.eye(3)
        bad[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            project_linear_weight(weight, bad, n_seen_classes=2)

    def test_dimension_mismatch_in_projection_raises(self):
        weight = torch.randn(2, 3)
        with self.assertRaises(ValueError):
            project_linear_weight(weight, torch.zeros(3, 4), n_seen_classes=2)

    def test_only_seen_rows_change_and_future_rows_are_bitwise_identical(self):
        generator = torch.Generator().manual_seed(19)
        weight = torch.randn(10, 512, generator=generator)
        projection = 0.5 * torch.eye(512)

        projected, stats = project_linear_weight(
            weight, projection, n_seen_classes=4,
        )

        self.assertFalse(torch.equal(projected[:4], weight[:4]))
        self.assertTrue(torch.equal(projected[4:], weight[4:]))
        self.assertEqual(stats['max_future_row_delta'], 0.0)
        self.assertEqual(stats['seen_weight_norm_before'], weight[:4].norm().item())

    def test_invalid_seen_class_count_raises(self):
        weight = torch.randn(3, 4)
        with self.assertRaises(ValueError):
            project_linear_weight(weight, torch.eye(4), n_seen_classes=0)
        with self.assertRaises(ValueError):
            project_linear_weight(weight, torch.eye(4), n_seen_classes=4)


class OracleScaleConsistencyTests(unittest.TestCase):
    """Sanity checks on the official SAP importance formula.

    The projection build code paths in utils/sap (conv) and models/dgc_sap
    (oracle Linear) both apply the same scaled-importance rule. The oracle
    branch exposes the same math via ``build_sap_projection_from_gram``.
    """

    def test_official_importance_formula_matches_alpha_r_over(self):
        generator = torch.Generator().manual_seed(7)
        # Construct a positive Gram so eigh recovers a meaningful spectrum.
        w = torch.randn(16, 6, generator=generator)
        gram = w.T @ w + 0.05 * torch.eye(6)
        eigvals = torch.linalg.eigvalsh(gram)

        for alpha in (1.0, 10.0, 100.0, 1000.0):
            r = eigvals / eigvals.sum()
            expected = alpha * r / ((alpha - 1.0) * r + 1.0)
            # Dominant direction (large r) importance approaches 1 for large α.
            self.assertLess(expected[-1].item(), 1.0 + 1e-6)
            # Weak direction (small r) importance approaches 0 for large α.
            self.assertGreater(expected[0].item(), 0.0)

    def test_build_sap_projection_from_gram_is_symmetric_and_psd_like(self):
        from utils.sap import build_sap_projection_from_gram

        generator = torch.Generator().manual_seed(13)
        w = torch.randn(10, 4, generator=generator)
        gram = w.T @ w

        projection = build_sap_projection_from_gram(gram, scale=100.0)

        # Symmetry: P == P.T
        torch.testing.assert_close(projection, projection.T, rtol=0, atol=1e-5)
        # All eigenvalues lie in [0, 1] (importance scaled per direction).
        eigvals = torch.linalg.eigvalsh(projection)
        self.assertGreaterEqual(eigvals.min().item(), -1e-5)
        self.assertLessEqual(eigvals.max().item(), 1.0 + 1e-5)

    def test_alpha_changes_projection_strength_but_not_experiment_command(self):
        from scripts.run_linear_sap_alpha_sweep import build_sweep_commands
        from utils.sap import build_sap_projection_from_gram

        base_command = [
            'python', 'main.py', '--dataset', 'seq-cifar10', '--model', 'dgc-sap',
            '--seed', '0', '--ogc_loss_weight', '0.3', '--sap_oracle_reference', '1',
        ]
        commands = build_sweep_commands(base_command, [100.0, 300.0])

        self.assertEqual(commands[0][:-2], base_command)
        self.assertEqual(commands[1][:-2], base_command)
        self.assertEqual(commands[0][-2:], ['--sap_oracle_scale', '100'])
        self.assertEqual(commands[1][-2:], ['--sap_oracle_scale', '300'])

        gram = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
        projection_100 = build_sap_projection_from_gram(gram, scale=100.0)
        projection_300 = build_sap_projection_from_gram(gram, scale=300.0)
        self.assertFalse(torch.allclose(projection_100, projection_300))


if __name__ == '__main__':
    unittest.main()
