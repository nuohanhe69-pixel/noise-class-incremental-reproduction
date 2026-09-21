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
    def test_task1_test_decision_capture_matches_seen_class_evaluation_and_state(self):
        from models.aer_sap import AerSap

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Identity()
                self.classifier = nn.Linear(2, 4)

            def forward(self, inputs):
                return self.classifier(self.backbone(inputs))

        class _Dataset:
            N_CLASSES = 4
            test_loaders = [[(
                torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
                torch.tensor([0, 1], dtype=torch.long),
            )]]

            @staticmethod
            def get_offsets(task_id=None):
                return 0, 2

        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.net = _Net()
        model.device = torch.device('cpu')
        model._current_task = 0
        model.args = SimpleNamespace(debug_mode=0)
        with torch.no_grad():
            model.net.classifier.weight.copy_(torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [100.0, 100.0],
                [50.0, 50.0],
            ]))
            model.net.classifier.bias.copy_(torch.tensor([0.1, 0.2, 10.0, 5.0]))

        state_before = {
            name: value.detach().clone() for name, value in model.net.state_dict().items()
        }
        training_before = {
            name: module.training for name, module in model.net.named_modules()
        }
        pre = model._capture_task1_test_decisions(_Dataset())

        torch.testing.assert_close(
            pre['raw_features'], torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
        )
        self.assertFalse(torch.allclose(
            pre['raw_features'].norm(dim=1), torch.ones(2),
        ))
        torch.testing.assert_close(
            pre['logits'],
            torch.nn.functional.linear(
                pre['raw_features'], state_before['classifier.weight'],
                state_before['classifier.bias'],
            ),
        )
        self.assertTrue(torch.equal(pre['predictions'], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(pre['logits'].argmax(dim=1), torch.tensor([2, 2])))
        self.assertEqual(pre['seen_classes'], [0, 1])
        self.assertEqual(pre['accuracy'], 100.0)
        for name, value in model.net.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]))
        self.assertEqual(
            {name: module.training for name, module in model.net.named_modules()},
            training_before,
        )

        weight_after = state_before['classifier.weight'].clone()
        weight_after[:2] = weight_after[:2].flip(0)
        model._install_classifier_candidate(
            model.net.classifier, weight_after, state_before['classifier.bias'],
        )
        post_state_before = {
            name: value.detach().clone() for name, value in model.net.state_dict().items()
        }
        post = model._capture_task1_test_decisions(_Dataset())
        torch.testing.assert_close(
            post['logits'],
            torch.nn.functional.linear(
                post['raw_features'], weight_after, state_before['classifier.bias'],
            ),
        )
        self.assertEqual(post['accuracy'], 0.0)
        for name, value in model.net.state_dict().items():
            self.assertTrue(torch.equal(value, post_state_before[name]))

        decision_stats = model._build_task1_test_decision_stats(
            pre,
            post,
            {
                'pre_sap': {'per_task_class_il': [100.0]},
                'post_sap': {'per_task_class_il': [0.0]},
            },
            classifier_has_bias=True,
        )
        self.assertEqual(decision_stats['pre_accuracy_absolute_error'], 0.0)
        self.assertEqual(decision_stats['post_accuracy_absolute_error'], 0.0)
        self.assertEqual(decision_stats['prediction_changed_count'], 2)

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

    def test_only_first_task_runs_taskwise_sap_after_every_aer_boundary(self):
        from models.aer_sap import AerSap, SAP_SKIPPED_FIRST_SESSION_ONLY

        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.sap_history = []
        dataset = SimpleNamespace(N_TASKS=10)
        boundary_order = []

        def finish_aer_boundary(instance, _dataset):
            boundary_order.append(('aer', instance.current_task))

        def run_first_session_sap(_dataset):
            boundary_order.append(('sap', model.current_task))

        with (
            patch.object(
                ErAceAerAbs, 'end_task', autospec=True,
                side_effect=finish_aer_boundary,
            ) as aer_end_task,
            patch.object(
                model, '_run_task_boundary_sap', side_effect=run_first_session_sap,
            ) as run_sap,
        ):
            for task_id in range(dataset.N_TASKS):
                model._current_task = task_id
                model.end_task(dataset)

        self.assertEqual(aer_end_task.call_count, dataset.N_TASKS)
        run_sap.assert_called_once_with(dataset)
        self.assertEqual(boundary_order[:2], [('aer', 0), ('sap', 0)])
        self.assertEqual(
            boundary_order[2:],
            [('aer', task_id) for task_id in range(1, dataset.N_TASKS)],
        )
        self.assertEqual(len(model.sap_history), dataset.N_TASKS - 1)
        self.assertTrue(all(
            event['status'] == SAP_SKIPPED_FIRST_SESSION_ONLY
            for event in model.sap_history
        ))
        self.assertEqual(
            [event['task_id'] for event in model.sap_history],
            list(range(1, dataset.N_TASKS)),
        )

    def test_first_session_taskwise_sap_projects_only_seen_rows_and_saves_task0(self):
        from models.aer_sap import AerSap

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(512, 512, bias=False)
                self.classifier = nn.Linear(512, 100)

            def forward(self, inputs):
                return self.classifier(self.backbone(inputs))

        class _Dataset:
            NAME = 'seq-cifar100'
            SETTING = 'class-il'
            N_TASKS = 10
            N_CLASSES = 100

            def __init__(self, classifier):
                self.classifier = classifier
                self.evaluated_weights = []
                self.test_loaders = [SimpleNamespace(dataset=range(20))]

            @staticmethod
            def get_offsets(task_id):
                return task_id * 10, (task_id + 1) * 10

            def evaluate(self, model, dataset):
                self.evaluated_weights.append(self.classifier.weight.detach().clone())
                branch = len(self.evaluated_weights)
                return [float(branch)], [float(branch + 10)]

        torch.manual_seed(4)
        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.net = _Net()
        model.device = torch.device('cpu')
        model._current_task = 0
        model._n_seen_classes = 10
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
        trusted_labels = torch.arange(10, dtype=torch.long)
        trusted_task_ids = torch.zeros(10, dtype=torch.long)
        reference_stats = {
            'current_task_clean_total': 10,
            'current_task_clean_count': 10,
            'current_task_clean_selected': 10,
            'buffer_clean_count': 3,
            'historical_buffer_clean_count': 0,
            'reference_new_count': 10,
            'reference_old_count': 0,
            'total_reference_count': 10,
            'current_class_selected_counts': {class_id: 1 for class_id in range(10)},
            'reference_sampling_seed': 0,
            'buffer_total_count': 3,
        }
        build_reference = Mock(return_value=(
            trusted_images, trusted_labels, trusted_task_ids, reference_stats,
        ))
        model._build_oracle_reference_batches = build_reference
        model._normalized_batches = Mock(return_value=iter([trusted_images.float()]))

        x_task1 = torch.zeros(10, 512)
        x_task1[torch.arange(10), torch.arange(10)] = 1.0
        feature_stats = {
            'before': {'min': 1.0, 'median': 1.0, 'mean': 1.0, 'max': 1.0},
            'after': {'min': 1.0, 'median': 1.0, 'mean': 1.0, 'max': 1.0},
        }
        def build_projection(gram):
            projection = torch.eye(512) * 2
            energy = torch.ones(512)
            normalized_energy = energy / energy.sum()
            importance = torch.ones(512) * 2
            return projection, energy, normalized_energy, importance

        model._build_oracle_projection = Mock(side_effect=build_projection)
        weight_before = model.net.classifier.weight.detach().clone()
        bias_before = model.net.classifier.bias.detach().clone()
        backbone_before = model.net.backbone.weight.detach().clone()
        task1_test_features_raw = torch.arange(
            100 * 512, dtype=torch.float32,
        ).reshape(100, 512) / 100
        task1_test_labels = torch.zeros(100, dtype=torch.long)
        pre_logits = torch.zeros(100, 100)
        pre_logits[:, 1] = 1.0
        pre_logits[0, 0] = 2.0
        post_logits = torch.zeros(100, 100)
        post_logits[:, 1] = 1.0
        post_logits[:2, 0] = 2.0
        pre_decisions = {
            'raw_features': task1_test_features_raw,
            'labels': task1_test_labels,
            'logits': pre_logits,
            'predictions': pre_logits[:, :10].argmax(dim=1),
            'seen_classes': list(range(10)),
            'accuracy': 1.0,
        }
        post_decisions = {
            'raw_features': task1_test_features_raw.clone(),
            'labels': task1_test_labels.clone(),
            'logits': post_logits,
            'predictions': post_logits[:, :10].argmax(dim=1),
            'seen_classes': list(range(10)),
            'accuracy': 2.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            'models.aer_sap.collect_classifier_input_features', return_value=x_task1,
        ) as collect_features, patch(
            'models.aer_sap.normalize_classifier_input_features',
            return_value=(x_task1, feature_stats),
        ) as normalize_features, patch.object(
            model, '_taskwise_artifact_directory', return_value=Path(temporary_directory),
        ), patch.object(
            model, '_capture_task1_test_decisions',
            side_effect=(pre_decisions, post_decisions),
        ) as capture_decisions:
            model._run_first_session_taskwise_sap(dataset)

            build_reference.assert_called_once_with(dataset, return_task_ids=True)
            collect_features.assert_called_once()
            normalize_features.assert_called_once_with(x_task1)
            self.assertEqual(capture_decisions.call_count, 2)
            self.assertEqual(model._build_oracle_projection.call_count, 1)
            self.assertEqual(len(dataset.evaluated_weights), 2)

            pre_sap_weight, post_sap_weight = dataset.evaluated_weights
            torch.testing.assert_close(pre_sap_weight, weight_before)
            torch.testing.assert_close(post_sap_weight[:10], weight_before[:10] * 2)
            self.assertTrue(torch.equal(post_sap_weight[10:], weight_before[10:]))

            torch.testing.assert_close(model.net.classifier.weight, post_sap_weight)
            self.assertTrue(torch.equal(model.net.classifier.bias, bias_before))
            self.assertTrue(torch.equal(model.net.backbone.weight, backbone_before))
            for name, value in model.net.state_dict().items():
                self.assertTrue(torch.equal(model.past_model_ckpt[name], value))

            artifact_dir = Path(temporary_directory)
            self.assertEqual(
                {
                    'W_before.pt', 'X_task1.pt', 'trusted_labels.pt',
                    'trusted_task_ids.pt', 'reference_stats.json', 'coverage.json',
                    'G_task_0.pt', 'M_task_0.pt', 'W_after.pt',
                    'task1_test_features_raw.pt', 'task1_test_labels.pt',
                    'classifier_bias_before.pt',
                    'task1_test_logits_pre_sap.pt',
                    'task1_test_logits_post_sap.pt',
                    'task1_test_predictions_pre_sap.pt',
                    'task1_test_predictions_post_sap.pt',
                    'task1_test_decision_stats.json',
                    'accuracy.json', 'manifest.json',
                },
                {path.name for path in artifact_dir.iterdir()},
            )
            saved_task_gram = torch.load(
                artifact_dir / 'G_task_0.pt', weights_only=True,
            )
            self.assertEqual(tuple(saved_task_gram.shape), (512, 512))
            torch.testing.assert_close(
                saved_task_gram, x_task1.transpose(0, 1) @ x_task1,
            )
            matrix = torch.load(
                artifact_dir / 'M_task_0.pt', weights_only=True,
            )
            self.assertEqual(tuple(matrix.shape), (512, 512))
            self.assertTrue(torch.equal(
                torch.load(artifact_dir / 'W_after.pt', weights_only=True),
                post_sap_weight,
            ))
            accuracy = json.loads((artifact_dir / 'accuracy.json').read_text())
            self.assertEqual(accuracy['pre_sap']['class_il'], 1.0)
            self.assertEqual(accuracy['pre_sap']['task_il'], 11.0)
            self.assertEqual(accuracy['post_sap']['class_il'], 2.0)
            self.assertEqual(accuracy['post_sap']['task_il'], 12.0)
            coverage = json.loads((artifact_dir / 'coverage.json').read_text())
            self.assertEqual(coverage['task_counts'], {'0': 10})
            self.assertEqual(set(coverage['tasks']), {'0'})
            self.assertEqual(coverage['empty_tasks'], [])
            saved_reference_stats = json.loads(
                (artifact_dir / 'reference_stats.json').read_text(),
            )
            self.assertEqual(saved_reference_stats['reference_new_count'], 10)
            self.assertEqual(saved_reference_stats['reference_old_count'], 0)
            manifest = json.loads((artifact_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['seen_tasks'], [0])
            self.assertFalse(manifest['global_candidate_built'])
            self.assertTrue(manifest['classifier_has_bias'])
            self.assertEqual(
                manifest['test_feature_type'], 'raw_classifier_input',
            )
            self.assertFalse(manifest['test_feature_l2_normalized'])
            torch.testing.assert_close(
                torch.load(
                    artifact_dir / 'task1_test_features_raw.pt', weights_only=True,
                ),
                task1_test_features_raw,
            )
            torch.testing.assert_close(
                torch.load(
                    artifact_dir / 'classifier_bias_before.pt', weights_only=True,
                ),
                bias_before,
            )
            decision_stats = json.loads(
                (artifact_dir / 'task1_test_decision_stats.json').read_text(),
            )
            self.assertEqual(decision_stats['sample_count'], 100)
            self.assertEqual(decision_stats['pre_accuracy_recomputed'], 1.0)
            self.assertEqual(decision_stats['post_accuracy_recomputed'], 2.0)
            self.assertEqual(decision_stats['pre_accuracy_absolute_error'], 0.0)
            self.assertEqual(decision_stats['post_accuracy_absolute_error'], 0.0)
            self.assertEqual(decision_stats['prediction_changed_count'], 1)
            self.assertEqual(model.sap_history[-1]['final_selected_candidate'], 'taskwise')
            self.assertEqual(model.sap_history[-1]['seen_tasks'], [0])
            self.assertEqual(
                model.sap_history[-1]['accuracy']['pre_sap']['class_il'], 1.0,
            )
            self.assertEqual(
                model.sap_history[-1]['accuracy']['post_sap']['class_il'], 2.0,
            )


if __name__ == '__main__':
    unittest.main()
