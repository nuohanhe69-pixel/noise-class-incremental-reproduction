import copy
import unittest

import torch
from torch import nn

from backbone.ResNetBlock import resnet18
from utils.sap import (
    RESNET18_LATE_STAGE_CONVS,
    apply_resnet18_sap_projections,
    collect_conv2d_input_gram,
    conv2d_input_to_patches,
    project_resnet18_from_reference_batches,
    resolve_resnet18_sap_layers,
)


class _TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Conv2d(1, 2, kernel_size=2, bias=False)

    def forward(self, inputs):
        return self.target(inputs)


class SAPResNetTests(unittest.TestCase):
    def test_resnet18_target_resolution_returns_only_the_ten_late_stage_convs(self):
        model = resnet18(num_classes=100, num_filters=2)

        layers = resolve_resnet18_sap_layers(model)

        self.assertEqual(tuple(layers), RESNET18_LATE_STAGE_CONVS)
        self.assertEqual(len(layers), 10)
        self.assertTrue(all(isinstance(layer, nn.Conv2d) for layer in layers.values()))

    def test_hook_gram_matches_all_direct_conv_input_patches(self):
        model = _TinyConvNet()
        inputs = torch.arange(27, dtype=torch.float32).reshape(3, 1, 3, 3)
        batches = [inputs[:2], inputs[2:]]
        expected_patches = conv2d_input_to_patches(inputs, model.target)

        stats = collect_conv2d_input_gram(
            model, batches, 'target', total_images=3, max_patches=100, seed=11,
        )

        torch.testing.assert_close(stats.gram, expected_patches.T @ expected_patches)
        self.assertEqual(stats.available_patches, 12)
        self.assertEqual(stats.sampled_patches, 12)
        self.assertEqual(stats.patch_dimension, 4)

    def test_capped_hook_sampling_is_deterministic_and_does_not_advance_global_rng(self):
        model = _TinyConvNet().train()
        model.target.eval()
        inputs = torch.arange(36, dtype=torch.float32).reshape(4, 1, 3, 3)
        parameters_before = copy.deepcopy(model.state_dict())
        global_rng_before = torch.random.get_rng_state().clone()

        first = collect_conv2d_input_gram(
            model, [inputs[:2], inputs[2:]], 'target',
            total_images=4, max_patches=7, seed=19,
        )
        second = collect_conv2d_input_gram(
            model, [inputs], 'target', total_images=4, max_patches=7, seed=19,
        )

        torch.testing.assert_close(first.gram, second.gram)
        self.assertEqual(first.sampled_patches, 7)
        self.assertEqual(first.available_patches, 16)
        self.assertTrue(model.training)
        self.assertFalse(model.target.training)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, parameters_before[name])

    def test_full_projection_uses_one_fresh_reference_pass_per_source_layer(self):
        source = resnet18(num_classes=100, num_filters=1)
        candidate = copy.deepcopy(source)
        source_before = copy.deepcopy(source.state_dict())
        candidate_before = copy.deepcopy(candidate.state_dict())
        inputs = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(23))
        factory_calls = 0

        def batch_factory():
            nonlocal factory_calls
            factory_calls += 1
            return [inputs[:1], inputs[1:]]

        stats = project_resnet18_from_reference_batches(
            source,
            candidate,
            batch_factory,
            total_images=2,
            max_patches=4,
            scale=10.0,
            seed=29,
        )

        self.assertEqual(factory_calls, 10)
        self.assertEqual(tuple(stats), RESNET18_LATE_STAGE_CONVS)
        self.assertTrue(all(layer_stats.sampled_patches == 4 for layer_stats in stats.values()))
        target_parameter_names = {f'{name}.weight' for name in RESNET18_LATE_STAGE_CONVS}
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, source_before[name])
        for name, value in candidate.state_dict().items():
            if name in target_parameter_names:
                self.assertFalse(torch.equal(value, candidate_before[name]), name)
            else:
                torch.testing.assert_close(value, candidate_before[name])

    def test_projection_changes_all_targets_and_no_other_candidate_state(self):
        source = resnet18(num_classes=100, num_filters=2)
        candidate = copy.deepcopy(source)
        target_layers = resolve_resnet18_sap_layers(candidate)
        projections = {
            name: torch.eye(layer.weight.flatten(1).shape[1]) * 0.5
            for name, layer in target_layers.items()
        }
        source_before = copy.deepcopy(source.state_dict())
        candidate_before = copy.deepcopy(candidate.state_dict())

        deltas = apply_resnet18_sap_projections(candidate, projections)

        self.assertEqual(tuple(deltas), RESNET18_LATE_STAGE_CONVS)
        self.assertTrue(all(delta > 0 for delta in deltas.values()))
        target_parameter_names = {f'{name}.weight' for name in RESNET18_LATE_STAGE_CONVS}
        for name, value in candidate.state_dict().items():
            if name in target_parameter_names:
                self.assertFalse(torch.equal(value, candidate_before[name]), name)
            else:
                torch.testing.assert_close(value, candidate_before[name])
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, source_before[name])

    def test_projection_rejects_non_target_layers(self):
        model = resnet18(num_classes=100, num_filters=2)
        projection = torch.eye(model.conv1.weight.flatten(1).shape[1])

        with self.assertRaisesRegex(ValueError, 'not an allowed SAP target'):
            apply_resnet18_sap_projections(model, {'conv1': projection})

    def test_projection_validates_every_matrix_before_changing_candidate(self):
        candidate = resnet18(num_classes=100, num_filters=2)
        layers = resolve_resnet18_sap_layers(candidate)
        projections = {
            name: torch.eye(layer.weight.flatten(1).shape[1]) * 0.5
            for name, layer in layers.items()
        }
        projections[RESNET18_LATE_STAGE_CONVS[-1]] = torch.eye(1)
        before = copy.deepcopy(candidate.state_dict())

        with self.assertRaisesRegex(ValueError, 'does not match'):
            apply_resnet18_sap_projections(candidate, projections)

        for name, value in candidate.state_dict().items():
            torch.testing.assert_close(value, before[name])


if __name__ == '__main__':
    unittest.main()
