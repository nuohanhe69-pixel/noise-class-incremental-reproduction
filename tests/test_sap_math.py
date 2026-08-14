import unittest

import torch
from torch import nn

from utils.sap import (
    build_sap_projection_from_gram,
    build_sap_projection_from_patches,
    conv2d_input_to_patches,
    project_conv2d_weight,
)


class SAPMathTests(unittest.TestCase):
    def test_conv2d_input_to_patches_uses_the_convolution_geometry(self):
        conv = nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
        inputs = torch.tensor([[[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0],
                                 [7.0, 8.0, 9.0]]]])

        patches = conv2d_input_to_patches(inputs, conv)

        expected = torch.tensor([
            [1.0, 2.0, 4.0, 5.0],
            [2.0, 3.0, 5.0, 6.0],
            [4.0, 5.0, 7.0, 8.0],
            [5.0, 6.0, 8.0, 9.0],
        ])
        torch.testing.assert_close(patches, expected)

    def test_gram_projection_matches_direct_svd_projection(self):
        generator = torch.Generator().manual_seed(7)
        patches = torch.randn(24, 6, generator=generator, dtype=torch.float64)

        direct = build_sap_projection_from_patches(patches, scale=3000.0)
        gram = build_sap_projection_from_gram(patches.T @ patches, scale=3000.0)

        torch.testing.assert_close(gram, direct, rtol=1e-9, atol=1e-10)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'requires Apple MPS')
    def test_mps_gram_projection_falls_back_to_cpu_eigendecomposition(self):
        gram = torch.diag(torch.tensor([9.0, 4.0, 1.0], device='mps'))

        projection = build_sap_projection_from_gram(gram, scale=3000.0)

        expected = build_sap_projection_from_gram(gram.cpu(), scale=3000.0)
        self.assertEqual(projection.device.type, 'mps')
        torch.testing.assert_close(projection.cpu(), expected, rtol=1e-5, atol=1e-6)

    def test_rank_limited_float32_gram_matches_svd_for_fewer_patches_than_dimensions(self):
        generator = torch.Generator().manual_seed(17)
        patches = torch.randn(4, 64, generator=generator)

        direct = build_sap_projection_from_patches(patches, scale=3000.0)
        gram = build_sap_projection_from_gram(
            patches.T @ patches, scale=3000.0, max_rank=patches.shape[0],
        )

        torch.testing.assert_close(gram, direct, rtol=2e-4, atol=2e-4)

    def test_projection_is_symmetric_and_has_bounded_eigenvalues(self):
        patches = torch.tensor([
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float64)

        projection = build_sap_projection_from_gram(patches.T @ patches, scale=10.0)
        eigenvalues = torch.linalg.eigvalsh(projection)

        torch.testing.assert_close(projection, projection.T)
        self.assertGreaterEqual(eigenvalues.min().item(), 0.0)
        self.assertLessEqual(eigenvalues.max().item(), 1.0)

    def test_projection_uses_the_official_singular_value_energy_scaling(self):
        patches = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
        scale = 10.0

        projection = build_sap_projection_from_patches(patches, scale=scale)

        energy_ratios = torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64) / 14.0
        expected_importance = scale * energy_ratios / ((scale - 1.0) * energy_ratios + 1.0)
        torch.testing.assert_close(torch.diag(projection), expected_importance)
        torch.testing.assert_close(projection, torch.diag(torch.diag(projection)))

    def test_projection_rejects_an_activation_space_with_zero_energy(self):
        with self.assertRaisesRegex(ValueError, 'positive activation energy'):
            build_sap_projection_from_gram(torch.zeros(3, 3), scale=3000.0)

    def test_conv2d_weight_projection_matches_the_official_pre_formula(self):
        weight = torch.arange(12, dtype=torch.float64).reshape(2, 1, 2, 3)
        projection = torch.eye(6, dtype=torch.float64)
        projection[0, 0] = 0.25
        projection[1, 1] = 0.50

        projected = project_conv2d_weight(weight, projection)
        expected = (weight.flatten(1) @ projection.T).reshape_as(weight)

        torch.testing.assert_close(projected, expected)
        self.assertEqual(projected.shape, weight.shape)
        torch.testing.assert_close(weight, torch.arange(12, dtype=torch.float64).reshape_as(weight))


if __name__ == '__main__':
    unittest.main()
