"""Targeted contracts for the E1 current-block hybrid offline evaluation."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import scripts.evaluate_e1_current_block_hybrid as e1
from utils.sap import project_linear_weight


class _NonUniformTaskDataset:
    N_TASKS = 3
    N_CLASSES = 7

    def __init__(self):
        self.requested_offsets = []

    def get_offsets(self, task_id):
        self.requested_offsets.append(task_id)
        return ((0, 2), (2, 5), (5, 7))[task_id]


class CurrentBlockHybridTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _NonUniformTaskDataset()
        self.weight_before = torch.arange(28, dtype=torch.float32).reshape(7, 4) + 1
        self.task_matrices = {
            0: torch.eye(4) * 2,
            1: torch.eye(4) * 3,
            2: torch.eye(4) * 4,
        }
        self.global_matrix = torch.eye(4) * 5

    def test_shared_old_local_base_is_built_once_and_candidates_only_change_current_rows(self):
        with patch.object(
            e1, 'project_linear_weight', wraps=project_linear_weight,
        ) as project:
            old_base, candidates = e1.build_hybrid_candidates(
                self.weight_before,
                self.task_matrices,
                self.global_matrix,
                self.dataset,
            )

        # Two old blocks are projected once total; Local and Global each project
        # the current block once. Current-Identity performs no current projection.
        self.assertEqual(project.call_count, 4)
        self.assertEqual(self.dataset.requested_offsets, [0, 1, 2])

        torch.testing.assert_close(old_base[0:2], self.weight_before[0:2] * 2)
        torch.testing.assert_close(old_base[2:5], self.weight_before[2:5] * 3)
        torch.testing.assert_close(old_base[5:7], self.weight_before[5:7])

        current_local = candidates['current_local']
        current_identity = candidates['current_identity']
        current_global = candidates['current_global']
        self.assertTrue(torch.equal(current_local[:5], current_identity[:5]))
        self.assertTrue(torch.equal(current_local[:5], current_global[:5]))
        torch.testing.assert_close(current_local[5:7], self.weight_before[5:7] * 4)
        self.assertTrue(torch.equal(current_identity[5:7], self.weight_before[5:7]))
        torch.testing.assert_close(current_global[5:7], self.weight_before[5:7] * 5)

        synthetic_saved_taskwise = self.weight_before.clone()
        synthetic_saved_taskwise[0:2] *= 2
        synthetic_saved_taskwise[2:5] *= 3
        synthetic_saved_taskwise[5:7] *= 4
        torch.testing.assert_close(current_local, synthetic_saved_taskwise)

        # Every result is an independent clone, not a view or a serial mutation.
        current_local[5, 0] = -999
        self.assertNotEqual(current_identity[5, 0].item(), -999)
        self.assertNotEqual(current_global[5, 0].item(), -999)
        self.assertNotEqual(old_base[5, 0].item(), -999)

    def test_candidate_contract_checks_old_rows_and_identity_current_rows_bitwise(self):
        _, candidates = e1.build_hybrid_candidates(
            self.weight_before,
            self.task_matrices,
            self.global_matrix,
            self.dataset,
        )

        offsets = e1.assert_candidate_row_contracts(
            candidates, self.weight_before, self.dataset,
        )

        self.assertEqual(offsets, (5, 7))

    def test_identity_weight_diagnostics_are_exact(self):
        current = self.weight_before[5:7]

        diagnostics = e1.compute_weight_diagnostics(current, current.clone())
        e1.assert_identity_diagnostics(diagnostics)

        self.assertEqual(diagnostics['relative_weight_delta'], 0.0)
        self.assertEqual(diagnostics['weight_norm_ratio'], 1.0)
        self.assertEqual(diagnostics['cosine'], 1.0)

    def test_identity_sanity_failure_stops_before_hybrid_evaluation(self):
        calls = []

        def evaluator(weight):
            calls.append(weight.clone())
            return {
                'class_il': 1.0,
                'task_il': 2.0,
                'per_task_class_il': [1.0],
                'per_task_task_il': [2.0],
            }

        expected = {
            'class_il': 99.0,
            'task_il': 2.0,
            'per_task_class_il': [99.0],
            'per_task_task_il': [2.0],
        }

        with self.assertRaises(AssertionError):
            e1.evaluate_identity_sanity(
                self.weight_before, expected, evaluator, tolerance=1e-4,
            )

        self.assertEqual(len(calls), 1)

    def test_post_evaluation_sanity_checks_old_task_il_and_current_identity(self):
        identity = {
            'per_task_task_il': [80.0, 81.0, 87.4],
        }
        accuracy = {
            'current_local': {'per_task_task_il': [80.0, 81.0, 10.8]},
            'current_identity': {'per_task_task_il': [80.0, 81.0, 87.4]},
            'current_global': {'per_task_task_il': [80.0, 81.0, 85.0]},
        }

        e1.assert_post_evaluation_contracts(
            identity, accuracy, last_task_id=2, tolerance=1e-6,
        )

        accuracy['current_global']['per_task_task_il'][0] = 79.0
        with self.assertRaises(AssertionError):
            e1.assert_post_evaluation_contracts(
                identity, accuracy, last_task_id=2, tolerance=1e-6,
            )

    def test_candidate_evaluation_restores_the_same_bias_before_every_evaluate(self):
        net = nn.Module()
        net.classifier = nn.Linear(3, 2)
        model = SimpleNamespace(net=net)
        bias_before = net.classifier.bias.detach().clone()
        evaluated_weights = []

        class _Dataset:
            def evaluate(self, candidate_model, dataset):
                torch.testing.assert_close(candidate_model.net.classifier.bias, bias_before)
                evaluated_weights.append(
                    candidate_model.net.classifier.weight.detach().clone(),
                )
                return [10.0, 20.0], [30.0, 40.0]

        first = torch.ones_like(net.classifier.weight)
        second = torch.full_like(net.classifier.weight, 2)
        net.classifier.bias.data.add_(7)

        first_result = e1.evaluate_weight_candidate(
            model, _Dataset(), net.classifier, first, bias_before,
        )
        net.classifier.bias.data.sub_(11)
        second_result = e1.evaluate_weight_candidate(
            model, _Dataset(), net.classifier, second, bias_before,
        )

        torch.testing.assert_close(evaluated_weights[0], first)
        torch.testing.assert_close(evaluated_weights[1], second)
        self.assertEqual(first_result['class_il'], 15.0)
        self.assertEqual(second_result['task_il'], 35.0)

    def test_source_provenance_is_read_from_checkpoint_args_and_fails_on_mismatch(self):
        args = SimpleNamespace(
            dataset='seq-cifar100', model='aer-sap', backbone='resnet18',
            seed=0, noise_rate=0.2, noise_type='symm', sap_oracle_scale=3000.0,
            debug_mode=0, eval_future=False,
        )

        provenance = e1.validate_source_provenance(args)
        self.assertEqual(provenance['dataset'], 'seq-cifar100')
        self.assertEqual(provenance['sap_oracle_scale'], 3000.0)

        args.seed = 1
        with self.assertRaises(ValueError):
            e1.validate_source_provenance(args)

    def test_source_provenance_accepts_symmetric_alias_and_rejects_other_noise_types(self):
        args = SimpleNamespace(
            dataset='seq-cifar100', model='aer-sap', backbone='resnet18',
            seed=0, noise_rate=0.2, noise_type='symmetric', sap_oracle_scale=3000.0,
            debug_mode=0, eval_future=False,
        )

        provenance = e1.validate_source_provenance(args)
        self.assertEqual(provenance['noise_type'], 'symmetric')

        args.noise_type = 'asymmetric'
        with self.assertRaises(ValueError):
            e1.validate_source_provenance(args)

    def test_output_directory_is_non_overwriting_and_artifacts_are_minimal(self):
        candidates = {
            'current_local': self.weight_before.clone(),
            'current_identity': self.weight_before.clone(),
            'current_global': self.weight_before.clone(),
        }
        accuracy = {'identity_sanity': {'class_il': 1.0}}
        diagnostics = {'current_identity': {'relative_weight_delta': 0.0}}
        manifest = {'experiment_name': 'e1_current_block_hybrid'}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'e1'
            e1.save_e1_artifacts(
                output, candidates, accuracy, diagnostics, manifest,
            )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    'W_current_local.pt',
                    'W_current_identity.pt',
                    'W_current_global.pt',
                    'accuracy.json',
                    'weight_diagnostics.json',
                    'experiment_manifest.json',
                },
            )
            self.assertEqual(
                json.loads((output / 'experiment_manifest.json').read_text()),
                manifest,
            )
            for filename in (
                'W_current_local.pt', 'W_current_identity.pt', 'W_current_global.pt',
            ):
                self.assertEqual(
                    torch.load(output / filename, weights_only=True).device.type,
                    'cpu',
                )

            with self.assertRaises(FileExistsError):
                e1.save_e1_artifacts(
                    output, candidates, accuracy, diagnostics, manifest,
                )

    def test_script_has_no_reference_feature_gram_or_projection_builder_calls(self):
        source = inspect.getsource(e1)
        forbidden_symbols = (
            '_build_oracle_reference_batches',
            'collect_classifier_input_features',
            'normalize_classifier_input_features',
            '_build_oracle_projection',
            'torch.linalg.eigh',
            'torch.linalg.svd',
        )

        for symbol in forbidden_symbols:
            self.assertNotIn(symbol, source)


if __name__ == '__main__':
    unittest.main()
