"""Synthetic contracts for the E3 Task10 Gram-centering evaluation."""

from __future__ import annotations

import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import scripts.evaluate_e3_task10_gram_centering as e3
from utils.sap import build_sap_projection_from_gram, project_linear_weight


class _TenTaskDataset:
    N_TASKS = 10

    @staticmethod
    def get_offsets(task_id):
        offsets = [
            (0, 1), (1, 3), (3, 4), (4, 6), (6, 7),
            (7, 9), (9, 10), (10, 12), (12, 13), (13, 15),
        ]
        return offsets[task_id]


class Task10CenteringTests(unittest.TestCase):
    @staticmethod
    def _accuracy(value):
        return {
            'class_il': float(value),
            'task_il': float(value + 1),
            'per_task_class_il': [float(value)] * 10,
            'per_task_task_il': [float(value + 1)] * 10,
        }

    def _make_orchestration_fixture(self, root):
        source = root / e3.SOURCE_RUN_ID
        source.mkdir()
        checkpoint = root / 'source_checkpoint.pt'
        checkpoint.write_bytes(b'synthetic checkpoint placeholder')

        dataset = _TenTaskDataset()
        weight_before = torch.arange(
            60, dtype=torch.float32,
        ).reshape(15, 4) / 10
        saved_projection = torch.eye(4) * 0.5
        checkpoint_weight = weight_before.clone()
        checkpoint_weight[13:15], _ = project_linear_weight(
            weight_before[13:15], saved_projection,
        )
        checkpoint_bias = torch.arange(15, dtype=torch.float32) / 100
        classifier = torch.nn.Linear(4, 15)
        with torch.no_grad():
            classifier.weight.copy_(checkpoint_weight)
            classifier.bias.copy_(checkpoint_bias)
        model = SimpleNamespace(net=SimpleNamespace())

        old_features = torch.eye(4)[torch.arange(9) % 4]
        task10_features = torch.eye(4)
        x_global = torch.cat((old_features, task10_features), dim=0)
        trusted_task_ids = torch.tensor([*range(9), 9, 9, 9, 9])
        saved_gram = task10_features.T @ task10_features
        source_accuracy = self._accuracy(20)

        tensors = {
            'W_before.pt': weight_before,
            'W_after_taskwise.pt': checkpoint_weight,
            'X_global.pt': x_global,
            'trusted_task_ids.pt': trusted_task_ids,
            'G_task_9.pt': saved_gram,
            'M_task_9.pt': saved_projection,
        }
        for filename, tensor in tensors.items():
            torch.save(tensor, source / filename)
        (source / 'accuracy.json').write_text(
            json.dumps({'taskwise': source_accuracy}), encoding='utf-8',
        )

        checkpoint_args = SimpleNamespace(
            dataset='seq-cifar100',
            model='aer-sap',
            backbone='resnet18',
            seed=0,
            noise_rate=0.2,
            noise_type='symmetric',
            sap_oracle_scale=3000.0,
            debug_mode=0,
            eval_future=False,
            device=None,
        )

        def load_checkpoint(_path, loaded_model=None, args=None, **kwargs):
            if kwargs.get('return_only_args'):
                return checkpoint_args
            self.assertIs(loaded_model, model)
            self.assertIsNone(args)
            return loaded_model, None

        return {
            'checkpoint': checkpoint,
            'source': source,
            'output': root / 'e3_output',
            'dataset': dataset,
            'model': model,
            'classifier': classifier,
            'checkpoint_args': checkpoint_args,
            'load_checkpoint': load_checkpoint,
            'weight_before': weight_before,
            'checkpoint_weight': checkpoint_weight,
            'checkpoint_bias': checkpoint_bias,
            'task10_features': task10_features,
            'source_accuracy': source_accuracy,
        }

    def _orchestration_patches(self, fixture, evaluator):
        return (
            patch.object(
                e3, 'initialize', return_value=(
                    fixture['model'], fixture['dataset'], fixture['checkpoint_args'],
                ),
            ),
            patch.object(
                e3, 'mammoth_load_checkpoint',
                side_effect=fixture['load_checkpoint'],
            ),
            patch.object(e3.e1, '_build_all_test_loaders'),
            patch.object(
                e3, 'resolve_classifier_module',
                return_value=fixture['classifier'],
            ),
            patch.object(
                e3.e1, 'evaluate_weight_candidate', side_effect=evaluator,
            ),
        )

    def test_task_layout_requires_exactly_ten_tasks_and_returns_zero_based_last_id(self):
        self.assertEqual(e3.validate_task_layout(_TenTaskDataset()), 9)

        class _WrongDataset:
            N_TASKS = 9

        with self.assertRaises(ValueError):
            e3.validate_task_layout(_WrongDataset())

    def test_task10_geometry_uses_saved_normalized_rows_and_centers_without_second_l2(self):
        raw = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0, 3.0],
            [3.0, 2.0, 4.0, 1.0],
            [1.0, 3.0, 4.0, 2.0],
        ])
        x_global = torch.nn.functional.normalize(raw, p=2, dim=1)
        task_ids = torch.tensor([8, 9, 7, 9, 9])
        expected_x10 = x_global[[1, 3, 4]]
        saved_uncentered_gram = expected_x10.T @ expected_x10

        geometry = e3.build_task10_centered_geometry(
            x_global,
            task_ids,
            last_task_id=9,
            saved_uncentered_gram=saved_uncentered_gram,
        )

        self.assertTrue(torch.equal(geometry['x_task'], expected_x10))
        expected_mean = expected_x10.mean(dim=0, keepdim=True)
        expected_centered = expected_x10 - expected_mean
        torch.testing.assert_close(geometry['mean'], expected_mean)
        torch.testing.assert_close(geometry['centered_features'], expected_centered)
        torch.testing.assert_close(
            geometry['centered_gram'], expected_centered.T @ expected_centered,
        )
        torch.testing.assert_close(
            geometry['centered_features'].mean(dim=0),
            torch.zeros(4),
            atol=1e-7,
            rtol=0,
        )
        self.assertFalse(torch.allclose(
            geometry['centered_features'].norm(dim=1),
            torch.ones(3),
        ))

        expected_shift = 3 * expected_mean.T @ expected_mean
        torch.testing.assert_close(
            saved_uncentered_gram - geometry['centered_gram'],
            expected_shift,
        )
        self.assertLess(geometry['diagnostics']['centered_mean_abs_max'], 1e-7)

    def test_task10_geometry_rejects_features_that_are_not_already_l2_normalized(self):
        x_global = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        task_ids = torch.tensor([9, 9])
        saved_uncentered_gram = x_global.T @ x_global

        with self.assertRaises(AssertionError):
            e3.build_task10_centered_geometry(
                x_global,
                task_ids,
                last_task_id=9,
                saved_uncentered_gram=saved_uncentered_gram,
            )

    def test_centered_projection_reuses_existing_helper_and_requires_alpha_3000(self):
        centered = torch.tensor([
            [1.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
        ])
        gram = centered.T @ centered

        with patch.object(
            e3,
            'build_sap_projection_from_gram',
            wraps=build_sap_projection_from_gram,
        ) as projection_builder:
            projection = e3.build_centered_projection(gram, scale=3000.0)

        self.assertEqual(projection_builder.call_count, 1)
        self.assertEqual(projection_builder.call_args.kwargs['scale'], 3000.0)
        self.assertEqual(tuple(projection.shape), (3, 3))

        with self.assertRaises(ValueError):
            e3.build_centered_projection(gram, scale=2999.0)

    def test_gram_spectrum_diagnostics_match_known_diagonal_spectrum(self):
        eigenvalues = torch.arange(12, 0, -1, dtype=torch.float64)
        diagnostics = e3._gram_spectrum_diagnostics(torch.diag(eigenvalues))
        probabilities = eigenvalues / eigenvalues.sum()
        expected_effective_rank = torch.exp(
            -(probabilities * probabilities.log()).sum(),
        ).item()

        self.assertEqual(
            set(diagnostics),
            {'trace', 'top1_energy_ratio', 'top10_energy_ratio', 'effective_rank'},
        )
        self.assertAlmostEqual(diagnostics['trace'], 78.0)
        self.assertAlmostEqual(diagnostics['top1_energy_ratio'], 12.0 / 78.0)
        self.assertAlmostEqual(diagnostics['top10_energy_ratio'], 75.0 / 78.0)
        self.assertAlmostEqual(
            diagnostics['effective_rank'], expected_effective_rank,
        )
        self.assertTrue(all(math.isfinite(value) for value in diagnostics.values()))

    def test_projection_diagnostics_match_known_diagonal_spectrum(self):
        projection = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        diagnostics = e3._projection_spectrum_diagnostics(projection)

        self.assertEqual(
            set(diagnostics),
            {
                'trace',
                'mean_eigenvalue',
                'median_eigenvalue',
                'frobenius_distance_to_identity_ratio',
            },
        )
        self.assertAlmostEqual(diagnostics['trace'], 6.0)
        self.assertAlmostEqual(diagnostics['mean_eigenvalue'], 2.0)
        self.assertAlmostEqual(diagnostics['median_eigenvalue'], 2.0)
        self.assertAlmostEqual(
            diagnostics['frobenius_distance_to_identity_ratio'],
            math.sqrt(5.0 / 3.0),
        )
        self.assertTrue(all(math.isfinite(value) for value in diagnostics.values()))

    def test_treatment_clones_control_and_replaces_only_last_task_from_w_before(self):
        dataset = _TenTaskDataset()
        weight_before = torch.arange(60, dtype=torch.float32).reshape(15, 4) + 1
        control_weight = weight_before + 1000
        centered_projection = torch.eye(4) * 0.25

        with patch.object(
            e3, 'project_linear_weight', wraps=project_linear_weight,
        ) as project:
            treatment, diagnostics = e3.build_centered_candidate(
                control_weight,
                weight_before,
                centered_projection,
                dataset,
                last_task_id=9,
            )

        self.assertEqual(project.call_count, 1)
        self.assertTrue(torch.equal(treatment[:13], control_weight[:13]))
        torch.testing.assert_close(treatment[13:15], weight_before[13:15] * 0.25)
        self.assertTrue(torch.equal(control_weight, weight_before + 1000))
        self.assertEqual(diagnostics['current_class_offsets'], [13, 15])
        self.assertTrue(diagnostics['old_rows_bitwise_equal'])

    def test_saved_uncentered_projection_reconstructs_control_task10(self):
        dataset = _TenTaskDataset()
        weight_before = torch.arange(60, dtype=torch.float32).reshape(15, 4) + 1
        saved_projection = torch.eye(4) * 0.5
        control_weight = weight_before.clone()
        control_weight[13:15], _ = project_linear_weight(
            weight_before[13:15], saved_projection,
        )

        diagnostics = e3.assert_control_task_reconstruction(
            control_weight,
            weight_before,
            saved_projection,
            dataset,
            last_task_id=9,
        )
        self.assertLessEqual(diagnostics['max_abs_delta'], 1e-7)

        control_weight[13, 0] += 1
        with self.assertRaises(AssertionError):
            e3.assert_control_task_reconstruction(
                control_weight,
                weight_before,
                saved_projection,
                dataset,
                last_task_id=9,
            )

    def test_artifacts_are_exact_cpu_set_and_output_is_non_overwriting(self):
        tensor = torch.eye(3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'e3'
            e3.save_e3_artifacts(
                output,
                task_mean=tensor[:1],
                centered_gram=tensor,
                centered_projection=tensor,
                centered_weight=tensor,
                accuracy={'current_uncentered_local': {'class_il': 1.0}},
                gram_diagnostics={'trace': 3.0},
                projection_diagnostics={'shape': [3, 3]},
                weight_diagnostics={'relative_weight_delta': 0.0},
                manifest={'experiment_name': 'e3_task10_gram_centering'},
            )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    'task10_mean.pt',
                    'G_task10_centered.pt',
                    'M_task10_centered.pt',
                    'W_current_centered.pt',
                    'accuracy.json',
                    'gram_diagnostics.json',
                    'projection_diagnostics.json',
                    'weight_diagnostics.json',
                    'experiment_manifest.json',
                },
            )
            self.assertEqual(
                json.loads((output / 'experiment_manifest.json').read_text()),
                {'experiment_name': 'e3_task10_gram_centering'},
            )
            for filename in (
                'task10_mean.pt',
                'G_task10_centered.pt',
                'M_task10_centered.pt',
                'W_current_centered.pt',
            ):
                self.assertEqual(
                    torch.load(output / filename, weights_only=True).device.type,
                    'cpu',
                )

            with self.assertRaises(FileExistsError):
                e3.save_e3_artifacts(
                    output,
                    task_mean=tensor[:1],
                    centered_gram=tensor,
                    centered_projection=tensor,
                    centered_weight=tensor,
                    accuracy={},
                    gram_diagnostics={},
                    projection_diagnostics={},
                    weight_diagnostics={},
                    manifest={},
                )

    def test_script_does_not_rebuild_reference_features_or_normalize_centered_rows(self):
        source = inspect.getsource(e3)
        forbidden_symbols = (
            '_build_oracle_reference_batches',
            'collect_classifier_input_features',
            'normalize_classifier_input_features',
            'torch.nn.functional.normalize',
            'torch.linalg.eigh',
            'torch.linalg.svd',
            'meta_begin_task',
            'meta_end_task',
        )
        for symbol in forbidden_symbols:
            self.assertNotIn(symbol, source)

    def test_run_e3_loads_source_then_evaluates_control_before_treatment_and_saves(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = self._make_orchestration_fixture(Path(temporary_directory))
            evaluated_weights = []
            treatment_accuracy = self._accuracy(30)

            def evaluate(model, dataset, classifier, candidate_weight, bias):
                self.assertIs(model, fixture['model'])
                self.assertIs(dataset, fixture['dataset'])
                e3.AerSap._install_classifier_candidate(
                    classifier, candidate_weight, bias,
                )
                evaluated_weights.append(candidate_weight.detach().clone())
                if len(evaluated_weights) == 1:
                    return fixture['source_accuracy']
                return treatment_accuracy

            patches = self._orchestration_patches(fixture, evaluate)
            with patches[0], patches[1] as checkpoint_loader, patches[2], \
                    patches[3], patches[4] as evaluator, patch.object(
                        e3,
                        'build_centered_projection',
                        wraps=e3.build_centered_projection,
                    ) as projection_builder:
                output = e3.run_e3(
                    fixture['checkpoint'], fixture['source'], fixture['output'],
                )

            self.assertEqual(checkpoint_loader.call_count, 2)
            self.assertTrue(
                checkpoint_loader.call_args_list[0].kwargs['return_only_args'],
            )
            self.assertEqual(evaluator.call_count, 2)
            self.assertTrue(torch.equal(
                evaluated_weights[0], fixture['checkpoint_weight'],
            ))
            self.assertTrue(torch.equal(
                evaluated_weights[1][:13], fixture['checkpoint_weight'][:13],
            ))
            self.assertFalse(torch.equal(
                evaluated_weights[1][13:15], fixture['checkpoint_weight'][13:15],
            ))
            self.assertEqual(
                projection_builder.call_args.kwargs['scale'], 3000.0,
            )
            self.assertEqual(output, fixture['output'].resolve())
            self.assertTrue((output / 'W_current_centered.pt').is_file())
            saved_accuracy = json.loads(
                (output / 'accuracy.json').read_text(encoding='utf-8'),
            )
            self.assertEqual(
                saved_accuracy['current_uncentered_local'],
                fixture['source_accuracy'],
            )
            self.assertEqual(
                saved_accuracy['current_centered_local'], treatment_accuracy,
            )
            torch.testing.assert_close(
                torch.load(output / 'task10_mean.pt', weights_only=True),
                fixture['task10_features'].mean(dim=0, keepdim=True),
            )
            for filename, expected_keys in (
                (
                    'gram_diagnostics.json',
                    {
                        'trace', 'top1_energy_ratio',
                        'top10_energy_ratio', 'effective_rank',
                    },
                ),
                (
                    'projection_diagnostics.json',
                    {
                        'trace', 'mean_eigenvalue', 'median_eigenvalue',
                        'frobenius_distance_to_identity_ratio',
                    },
                ),
            ):
                diagnostics = json.loads(
                    (output / filename).read_text(encoding='utf-8'),
                )
                for state in ('uncentered', 'centered'):
                    self.assertEqual(
                        set(diagnostics[state]), expected_keys,
                    )
                    self.assertTrue(all(
                        math.isfinite(value)
                        for value in diagnostics[state].values()
                    ))

    def test_run_e3_control_accuracy_mismatch_stops_before_treatment_and_save(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = self._make_orchestration_fixture(Path(temporary_directory))
            evaluated_weights = []

            def mismatched_control(model, dataset, classifier, candidate_weight, bias):
                e3.AerSap._install_classifier_candidate(
                    classifier, candidate_weight, bias,
                )
                evaluated_weights.append(candidate_weight.detach().clone())
                return self._accuracy(999)

            patches = self._orchestration_patches(fixture, mismatched_control)
            with patches[0], patches[1], patches[2], patches[3], \
                    patches[4] as evaluator, patch.object(
                        e3, 'save_e3_artifacts', wraps=e3.save_e3_artifacts,
                    ) as artifact_saver:
                with self.assertRaises(AssertionError):
                    e3.run_e3(
                        fixture['checkpoint'],
                        fixture['source'],
                        fixture['output'],
                    )

            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(len(evaluated_weights), 1)
            self.assertTrue(torch.equal(
                evaluated_weights[0], fixture['checkpoint_weight'],
            ))
            artifact_saver.assert_not_called()
            self.assertFalse(fixture['output'].exists())

    def test_run_e3_restores_classifier_when_treatment_evaluation_raises(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = self._make_orchestration_fixture(Path(temporary_directory))
            original_weight = fixture['classifier'].weight.detach().clone()
            original_bias = fixture['classifier'].bias.detach().clone()
            evaluation_count = 0

            def fail_treatment(model, dataset, classifier, candidate_weight, bias):
                nonlocal evaluation_count
                evaluation_count += 1
                e3.AerSap._install_classifier_candidate(
                    classifier, candidate_weight, bias,
                )
                if evaluation_count == 1:
                    return fixture['source_accuracy']
                raise RuntimeError('synthetic treatment evaluation failure')

            patches = self._orchestration_patches(fixture, fail_treatment)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(RuntimeError, 'treatment evaluation'):
                    e3.run_e3(
                        fixture['checkpoint'],
                        fixture['source'],
                        fixture['output'],
                    )

            self.assertEqual(evaluation_count, 2)
            self.assertTrue(torch.equal(
                fixture['classifier'].weight, original_weight,
            ))
            self.assertTrue(torch.equal(
                fixture['classifier'].bias, original_bias,
            ))
            self.assertFalse(fixture['output'].exists())

    def test_run_e3_restores_classifier_when_artifact_save_raises(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = self._make_orchestration_fixture(Path(temporary_directory))
            original_weight = fixture['classifier'].weight.detach().clone()
            original_bias = fixture['classifier'].bias.detach().clone()
            evaluation_count = 0

            def evaluate(model, dataset, classifier, candidate_weight, bias):
                nonlocal evaluation_count
                evaluation_count += 1
                e3.AerSap._install_classifier_candidate(
                    classifier, candidate_weight, bias,
                )
                if evaluation_count == 1:
                    return fixture['source_accuracy']
                return self._accuracy(30)

            patches = self._orchestration_patches(fixture, evaluate)
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patch.object(
                        e3,
                        'save_e3_artifacts',
                        side_effect=OSError('synthetic artifact save failure'),
                    ):
                with self.assertRaisesRegex(OSError, 'artifact save'):
                    e3.run_e3(
                        fixture['checkpoint'],
                        fixture['source'],
                        fixture['output'],
                    )

            self.assertEqual(evaluation_count, 2)
            self.assertTrue(torch.equal(
                fixture['classifier'].weight, original_weight,
            ))
            self.assertTrue(torch.equal(
                fixture['classifier'].bias, original_bias,
            ))
            self.assertFalse(fixture['output'].exists())


if __name__ == '__main__':
    unittest.main()
