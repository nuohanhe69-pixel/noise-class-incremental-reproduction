"""Contract tests for the pure AER + Oracle Linear SAP ablation entry."""

import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from backbone.ResNetBlock import resnet18
from models import get_model_names
from models.dgc import DGC
from models.dgc_sap import DgcSap, SAP_ORACLE_EXECUTED
from models.er_ace_aer_abs import ErAceAerAbs
from tests.test_sap_oracle_smoke import _FakeTrainDataset
from utils.buffer import Buffer


class AerSapContractTests(unittest.TestCase):
    def test_inherits_aer_directly_without_dgc(self):
        from models.aer_sap import AerSap

        self.assertIs(AerSap.__bases__[0], ErAceAerAbs)
        self.assertNotIn(DGC, inspect.getmro(AerSap))

    def test_parser_has_oracle_sap_but_no_ogc_arguments(self):
        from models.aer_sap import AerSap

        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        destinations = {action.dest for action in parser._actions}

        self.assertIn('sap_oracle_reference', destinations)
        self.assertIn('sap_oracle_scale', destinations)
        self.assertIn('sap_batch_size', destinations)
        self.assertFalse(any(name.startswith('ogc_') for name in destinations))

    def test_reuses_exact_dgc_sap_oracle_pipeline(self):
        from models.aer_sap import AerSap

        shared_methods = (
            '_normalized_batches',
            '_build_oracle_reference_batches',
            '_run_oracle_classifier_sap',
            '_evaluate_seen_task_accuracies',
            '_build_oracle_projection',
            '_sap_importance_from_energy',
        )
        for method_name in shared_methods:
            self.assertIs(
                getattr(AerSap, method_name),
                getattr(DgcSap, method_name),
                msg=f'{method_name} must stay identical to dgc-sap oracle mode',
            )

    def test_model_registry_recognizes_hyphenated_cli_name(self):
        from models.aer_sap import AerSap

        self.assertIs(get_model_names()['aer-sap'], AerSap)

    def test_enables_oracle_buffer_metadata_without_enabling_loss_trace(self):
        from models.aer_sap import AerSap

        model = AerSap.__new__(AerSap)
        model.loss_trace_recorder = None

        self.assertTrue(model._should_store_buffer_metadata())
        self.assertIsNone(model.loss_trace_recorder)

        labels = torch.tensor([2, 3, 2])
        model._current_task = 4
        source_task_ids = model._buffer_source_task_ids(None, labels)
        torch.testing.assert_close(source_task_ids, torch.tensor([4, 4, 4]))

    def test_oracle_boundary_projects_full_linear_and_refreshes_aer_checkpoint(self):
        from argparse import Namespace
        from models.aer_sap import AerSap

        true_labels = torch.arange(16, dtype=torch.long) % 2
        observed_labels = true_labels.clone()
        observed_labels[::4] = 1 - observed_labels[::4]
        train_dataset = _FakeTrainDataset(16, 2, true_labels, observed_labels)

        class _Loader:
            dataset = train_dataset

        class _Dataset:
            train_loader = _Loader()
            test_loaders = [[(
                torch.rand(8, 3, 32, 32),
                torch.arange(8, dtype=torch.long) % 2,
            )]]

        backbone = resnet18(num_classes=10, num_filters=4)
        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.net = backbone
        model.device = torch.device('cpu')
        model._current_task = 0
        model._n_classes_current_task = 2
        model._n_seen_classes = 2
        model.buffer = Buffer(4, torch.device('cpu'), sample_selection_strategy='reservoir')
        model.sap_history = []
        model.args = Namespace(sap_oracle_scale=100.0, sap_batch_size=4)

        def batches(images, labels=None):
            tensor = torch.as_tensor(images).float() / 255
            if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
            for start in range(0, len(tensor), 4):
                batch = tensor[start:start + 4]
                if labels is None:
                    yield batch
                else:
                    yield batch, torch.as_tensor(labels[start:start + 4]).long()

        model._normalized_batches = batches
        weight_before = backbone.classifier.weight.detach().clone()
        bias_before = backbone.classifier.bias.detach().clone()

        model._run_oracle_classifier_sap(_Dataset())

        self.assertFalse(torch.equal(weight_before, backbone.classifier.weight))
        self.assertFalse(torch.equal(weight_before[2:], backbone.classifier.weight[2:]))
        torch.testing.assert_close(bias_before, backbone.classifier.bias)
        self.assertEqual(model.sap_history[-1]['status'], SAP_ORACLE_EXECUTED)
        self.assertEqual(model.sap_history[-1]['total_reference_count'], 12)
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(model.past_model_ckpt[name], value)

    def test_only_final_task_runs_taskwise_sap_after_every_aer_boundary(self):
        from models.aer_sap import AerSap, SAP_SKIPPED_FINAL_ONLY

        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.sap_history = []
        dataset = SimpleNamespace(N_TASKS=10)

        with (
            patch.object(ErAceAerAbs, 'end_task', autospec=True) as aer_end_task,
            patch.object(model, '_run_task_boundary_sap') as run_sap,
        ):
            for task_id in range(dataset.N_TASKS):
                model._current_task = task_id
                model.end_task(dataset)

        self.assertEqual(aer_end_task.call_count, dataset.N_TASKS)
        run_sap.assert_called_once_with(dataset)
        self.assertEqual(len(model.sap_history), dataset.N_TASKS - 1)
        self.assertTrue(all(
            event['status'] == SAP_SKIPPED_FINAL_ONLY
            for event in model.sap_history
        ))

    def test_final_taskwise_sap_uses_one_reference_and_one_feature_matrix(self):
        from models.aer_sap import AerSap

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(512, 100)

            def forward(self, inputs):
                return self.classifier(inputs)

        class _Dataset:
            NAME = 'seq-cifar100'
            SETTING = 'class-il'
            N_TASKS = 10
            N_CLASSES = 100

            def __init__(self, classifier):
                self.classifier = classifier
                self.evaluated_weights = []
                self.test_loaders = []

            @staticmethod
            def get_offsets(task_id):
                return task_id * 10, (task_id + 1) * 10

            def evaluate(self, model, dataset):
                self.evaluated_weights.append(self.classifier.weight.detach().clone())
                branch = len(self.evaluated_weights)
                return [float(branch)] * 10, [float(branch + 10)] * 10

        torch.manual_seed(4)
        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.net = _Net()
        model.device = torch.device('cpu')
        model._current_task = 9
        model._n_seen_classes = 100
        model.sap_history = []
        model.args = SimpleNamespace(
            sap_oracle_scale=3000.0,
            sap_batch_size=32,
            seed=0,
            results_path='results',
            conf_jobnum='test-run',
        )
        dataset = _Dataset(model.net.classifier)

        trusted_images = torch.zeros(10, 3, 32, 32, dtype=torch.uint8)
        trusted_labels = torch.arange(10, dtype=torch.long) * 10
        trusted_task_ids = torch.arange(10, dtype=torch.long)
        reference_stats = {
            'current_task_clean_total': 5,
            'current_task_clean_count': 5,
            'current_task_clean_selected': 5,
            'buffer_clean_count': 5,
            'historical_buffer_clean_count': 5,
            'reference_new_count': 5,
            'reference_old_count': 5,
            'total_reference_count': 10,
            'current_class_selected_counts': {90: 5},
            'reference_sampling_seed': 9,
            'buffer_total_count': 5,
        }
        build_reference = Mock(return_value=(
            trusted_images, trusted_labels, trusted_task_ids, reference_stats,
        ))
        model._build_oracle_reference_batches = build_reference
        model._normalized_batches = Mock(return_value=iter([trusted_images.float()]))

        x_global = torch.zeros(10, 512)
        x_global[torch.arange(10), torch.arange(10)] = 1.0
        feature_stats = {
            'before': {'min': 1.0, 'median': 1.0, 'mean': 1.0, 'max': 1.0},
            'after': {'min': 1.0, 'median': 1.0, 'mean': 1.0, 'max': 1.0},
        }
        projection_call = 0

        def build_projection(gram):
            nonlocal projection_call
            factor = float(projection_call + 2)
            projection_call += 1
            projection = torch.eye(512) * factor
            energy = torch.ones(512)
            normalized_energy = energy / energy.sum()
            importance = torch.ones(512) * factor
            return projection, energy, normalized_energy, importance

        model._build_oracle_projection = Mock(side_effect=build_projection)
        weight_before = model.net.classifier.weight.detach().clone()
        bias_before = model.net.classifier.bias.detach().clone()

        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            'models.aer_sap.collect_classifier_input_features', return_value=x_global,
        ) as collect_features, patch(
            'models.aer_sap.normalize_classifier_input_features',
            return_value=(x_global, feature_stats),
        ) as normalize_features, patch.object(
            model, '_taskwise_artifact_directory', return_value=Path(temporary_directory),
        ):
            model._run_final_taskwise_sap(dataset)

            build_reference.assert_called_once_with(dataset, return_task_ids=True)
            collect_features.assert_called_once()
            normalize_features.assert_called_once_with(x_global)
            self.assertEqual(model._build_oracle_projection.call_count, 11)
            self.assertEqual(len(dataset.evaluated_weights), 3)

            identity_weight, global_weight, taskwise_weight = dataset.evaluated_weights
            torch.testing.assert_close(identity_weight, weight_before)
            torch.testing.assert_close(global_weight, weight_before * 2)
            for task_id in range(10):
                start_c, end_c = dataset.get_offsets(task_id)
                torch.testing.assert_close(
                    taskwise_weight[start_c:end_c],
                    weight_before[start_c:end_c] * (task_id + 3),
                )

            torch.testing.assert_close(model.net.classifier.weight, taskwise_weight)
            torch.testing.assert_close(model.net.classifier.bias, bias_before)
            for name, value in model.net.state_dict().items():
                torch.testing.assert_close(model.past_model_ckpt[name], value)

            artifact_dir = Path(temporary_directory)
            self.assertEqual(
                {
                    'W_before.pt', 'X_global.pt', 'trusted_labels.pt',
                    'trusted_task_ids.pt', 'coverage.json', 'G_global.pt',
                    'M_global.pt',
                    'W_after_global.pt', 'W_after_taskwise.pt', 'accuracy.json',
                    *(f'G_task_{task_id}.pt' for task_id in range(10)),
                    *(f'M_task_{task_id}.pt' for task_id in range(10)),
                },
                {path.name for path in artifact_dir.iterdir()},
            )
            saved_global_gram = torch.load(
                artifact_dir / 'G_global.pt', weights_only=True,
            )
            self.assertEqual(tuple(saved_global_gram.shape), (512, 512))
            torch.testing.assert_close(
                saved_global_gram, x_global.transpose(0, 1) @ x_global,
            )
            for task_id in range(10):
                task_features = x_global[trusted_task_ids == task_id]
                saved_task_gram = torch.load(
                    artifact_dir / f'G_task_{task_id}.pt', weights_only=True,
                )
                self.assertEqual(tuple(saved_task_gram.shape), (512, 512))
                torch.testing.assert_close(
                    saved_task_gram,
                    task_features.transpose(0, 1) @ task_features,
                )
                matrix = torch.load(
                    artifact_dir / f'M_task_{task_id}.pt', weights_only=True,
                )
                self.assertEqual(tuple(matrix.shape), (512, 512))
            accuracy = json.loads((artifact_dir / 'accuracy.json').read_text())
            self.assertEqual(accuracy['identity']['class_il'], 1.0)
            self.assertEqual(accuracy['global']['class_il'], 2.0)
            self.assertEqual(accuracy['taskwise']['class_il'], 3.0)
            coverage = json.loads((artifact_dir / 'coverage.json').read_text())
            self.assertEqual(coverage['task_counts'], {str(i): 1 for i in range(10)})
            self.assertEqual(coverage['empty_tasks'], [])
            self.assertEqual(model.sap_history[-1]['final_selected_candidate'], 'taskwise')


if __name__ == '__main__':
    unittest.main()
