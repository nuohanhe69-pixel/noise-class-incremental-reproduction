"""Focused contracts for the two AER/ABS task-boundary SAP operations."""

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from models.aer_sap import AerSap, SAP_SKIPPED_AFTER_SECOND_BOUNDARY
from models.dgc_sap import DgcSap, SAP_FAILED, SAP_ORACLE_EXECUTED
from models.er_ace_aer_abs import ErAceAerAbs


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4, bias=False)
        self.classifier = nn.Linear(4, 100)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(4))

    def forward(self, inputs):
        return self.classifier(self.backbone(inputs))


class _Loader:
    def __init__(self, inputs, labels, indexes):
        self.batches = [(inputs, labels)]
        self.dataset = SimpleNamespace(indexes=indexes)

    def __iter__(self):
        return iter(self.batches)


class _Dataset:
    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_TASKS = 10
    N_CLASSES = 100

    def __init__(self):
        self.test_loaders = [
            _Loader(torch.tensor([[2., 0., 0., 0.], [0., 2., 0., 0.]]),
                    torch.tensor([0, 1]), [100, 101]),
            _Loader(torch.tensor([[0., 0., 2., 0.], [0., 0., 0., 2.]]),
                    torch.tensor([10, 11]), [200, 201]),
        ]

    @staticmethod
    def get_offsets(task_id):
        return task_id * 10, (task_id + 1) * 10

    def evaluate(self, model, _dataset):
        seen_end = (model.current_task + 1) * 10
        class_accs, task_accs = [], []
        with torch.no_grad():
            for task_id in range(model.current_task + 1):
                inputs, labels = self.test_loaders[task_id].batches[0]
                logits = model.net(inputs)
                class_accs.append(float((logits[:, :seen_end].argmax(1) == labels).float().mean() * 100))
                start, end = self.get_offsets(task_id)
                task_accs.append(float((logits[:, start:end].argmax(1) + start == labels).float().mean() * 100))
        return class_accs, task_accs


def _reference(task_id):
    if task_id == 0:
        images = torch.tensor([[2., 0., 0., 0.], [0., 2., 0., 0.]])
        labels = torch.tensor([0, 1])
        ids = torch.tensor([0, 1])
        tasks = torch.zeros(2, dtype=torch.long)
        new_count = 2
    else:
        images = torch.tensor([
            [0., 0., 255., 0.], [0., 0., 0., 255.],
            [2., 0., 0., 0.], [0., 2., 0., 0.],
        ])
        labels = torch.tensor([10, 11, 0, 1])
        ids = torch.tensor([10, 11, 0, 1])
        tasks = torch.tensor([1, 1, 0, 0])
        new_count = 2
    counts = {
        'current_candidate': 2, 'current_eligible': 2, 'current_selected': 2,
        'old_candidate': task_id * 2, 'old_eligible': task_id * 2,
        'old_selected': task_id * 2, 'new_candidate': 2,
        'new_eligible': 2, 'new_selected': 2,
        'selected_per_class': {int(label): 1 for label in labels},
        'sampling_seed': task_id,
    }
    evidence = {
        'sample_ids': ids, 'true_labels': labels.clone(),
        'observed_labels': labels.clone(), 'source_task_ids': tasks.clone(),
        'is_old': torch.arange(len(labels)) >= new_count,
        'reference_order': torch.arange(len(labels)), 'counts': counts,
    }
    return images, labels, tasks, {'reference_new_count': 2,
                                    'reference_old_count': task_id * 2}, evidence


def _model(results_path):
    model = AerSap.__new__(AerSap)
    nn.Module.__init__(model)
    model.net = _Net()
    model.device = torch.device('cpu')
    model._current_task = 0
    model._n_seen_classes = 10
    model.args = SimpleNamespace(
        sap_oracle_scale=300.0, sap_batch_size=4, seed=0,
        results_path=str(results_path), conf_jobnum='contract',
        debug_mode=False, non_verbose=True,
    )
    model.normalization_transform = nn.Identity()
    model.sap_history = []
    model._normalized_batches = lambda images, labels: iter([(images, labels)])
    model._build_oracle_reference_batches = Mock(
        side_effect=lambda dataset, **kwargs: _reference(model.current_task)
    )
    return model


class AerSapContractTests(unittest.TestCase):
    def test_parser_and_reused_math(self):
        self.assertIs(AerSap.__bases__[0], ErAceAerAbs)
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        self.assertEqual(parser.parse_args(['--buffer_size', '20']).sap_oracle_scale, 300.0)
        self.assertIs(AerSap._build_oracle_projection, DgcSap._build_oracle_projection)
        self.assertIs(AerSap._build_oracle_reference_batches, DgcSap._build_oracle_reference_batches)

    def test_task0_to_9_schedule_after_aer_boundary(self):
        model = _model('unused')
        dataset = _Dataset()
        order = []
        with patch.object(ErAceAerAbs, 'end_task', autospec=True,
                          side_effect=lambda instance, ds: order.append(('aer', instance.current_task))), \
             patch.object(model, '_run_task_boundary_sap',
                          side_effect=lambda ds: order.append(('sap', model.current_task))):
            for task_id in range(10):
                model._current_task = task_id
                model.end_task(dataset)
        self.assertEqual([task for kind, task in order if kind == 'sap'], [0, 1])
        self.assertEqual(order[:4], [('aer', 0), ('sap', 0), ('aer', 1), ('sap', 1)])
        self.assertEqual([event['status'] for event in model.sap_history],
                         [SAP_SKIPPED_AFTER_SECOND_BOUNDARY] * 8)

    def test_two_boundaries_save_independent_reconstructable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(tmp)
            dataset = _Dataset()
            model._run_taskwise_sap(dataset)
            first = model._taskwise_artifact_directory(dataset)
            first_manifest = json.loads((first / 'manifest.json').read_text())
            first_matrix = torch.load(first / 'M_task_0.pt', weights_only=True)
            self.assertEqual(first_manifest['sap_alpha'], 300.0)
            self.assertEqual(first_manifest['seen_tasks'], [0])

            with torch.no_grad():
                model.net.backbone.weight[0, 1] += 0.5
            model._current_task = 1
            model._n_seen_classes = 20
            pre_state = copy.deepcopy(model.net.state_dict())
            model._run_taskwise_sap(dataset)
            second = model._taskwise_artifact_directory(dataset)
            self.assertNotEqual(first, second)
            self.assertTrue((first / 'manifest.json').exists())
            manifest = json.loads((second / 'manifest.json').read_text())
            self.assertEqual(manifest['seen_tasks'], [0, 1])
            self.assertTrue(manifest['valid'] and manifest['artifact_complete'])
            self.assertEqual(manifest['reference_count'], 4)
            self.assertEqual(set(manifest['files']),
                             {path.name for path in second.iterdir()} - {'manifest.json'})
            self.assertFalse(torch.equal(first_matrix,
                                         torch.load(second / 'M_task_0.pt', weights_only=True)))

            before = torch.load(second / 'W_before.pt', weights_only=True)
            after = torch.load(second / 'W_after.pt', weights_only=True)
            x_raw = torch.load(second / 'X_raw.pt', weights_only=True)
            expected_new = torch.eye(4)[2:4] @ pre_state['backbone.weight'].T
            torch.testing.assert_close(x_raw[:2], expected_new)
            self.assertTrue(torch.equal(before[20:], after[20:]))
            for task_id in (0, 1):
                x = torch.load(second / f'X_l2_task_{task_id}.pt', weights_only=True)
                gram = torch.load(second / f'G_task_{task_id}.pt', weights_only=True)
                matrix = torch.load(second / f'M_task_{task_id}.pt', weights_only=True)
                eigenvalues = torch.load(second / f'eigenvalues_task_{task_id}.pt', weights_only=True)
                eigenvectors = torch.load(second / f'eigenvectors_task_{task_id}.pt', weights_only=True)
                importance = torch.load(second / f'importance_task_{task_id}.pt', weights_only=True)
                torch.testing.assert_close(gram, x.T @ x)
                torch.testing.assert_close(gram, (eigenvectors * eigenvalues) @ eigenvectors.T)
                torch.testing.assert_close(matrix, (eigenvectors * importance) @ eigenvectors.T)
                self.assertEqual(eigenvalues.shape, (4,))
                start, end = dataset.get_offsets(task_id)
                torch.testing.assert_close(after[start:end], before[start:end] @ matrix.T)
            self.assertTrue(torch.equal(
                torch.load(second / 'bias_before.pt', weights_only=True),
                torch.load(second / 'bias_after.pt', weights_only=True),
            ))
            post_state = torch.load(second / 'network_post_sap.pt', weights_only=True)
            for key, value in pre_state.items():
                if key != 'classifier.weight':
                    self.assertTrue(torch.equal(value, post_state[key]))
                self.assertTrue(torch.equal(model.past_model_ckpt[key], model.net.state_dict()[key]))

            for phase in ('pre_sap', 'post_sap'):
                sample_ids = torch.load(second / f'test_{phase}_sample_ids.pt', weights_only=True)
                task_ids = torch.load(second / f'test_{phase}_task_ids.pt', weights_only=True)
                labels = torch.load(second / f'test_{phase}_labels.pt', weights_only=True)
                logits = torch.load(second / f'test_{phase}_logits.pt', weights_only=True)
                predictions = torch.load(second / f'test_{phase}_predictions.pt', weights_only=True)
                self.assertEqual(sample_ids.tolist(), [100, 101, 200, 201])
                self.assertEqual(task_ids.tolist(), [0, 0, 1, 1])
                self.assertEqual(labels.tolist(), [0, 1, 10, 11])
                self.assertTrue(torch.equal(predictions, logits[:, :20].argmax(1)))
                accuracy = json.loads((second / 'accuracy.json').read_text())[phase]
                for task_id in (0, 1):
                    mask = task_ids == task_id
                    expected = float((predictions[mask] == labels[mask]).float().mean() * 100)
                    self.assertEqual(expected, accuracy['per_task_class_il'][task_id])
            for key in ('sample_ids', 'task_ids', 'labels', 'raw_features'):
                self.assertTrue(torch.equal(
                    torch.load(second / f'test_pre_sap_{key}.pt', weights_only=True),
                    torch.load(second / f'test_post_sap_{key}.pt', weights_only=True),
                ))
            self.assertEqual([event['sap_alpha'] for event in model.sap_history
                              if event['status'] == SAP_ORACLE_EXECUTED], [300.0, 300.0])

    def test_save_failure_rolls_back_network_and_aer_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(tmp)
            dataset = _Dataset()
            model.past_model_ckpt = copy.deepcopy(model.net.state_dict())
            before = copy.deepcopy(model.net.state_dict())
            past_before = copy.deepcopy(model.past_model_ckpt)
            with patch.object(model, '_save_taskwise_artifacts', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    model._run_taskwise_sap(dataset)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.net.state_dict()[key]))
                self.assertTrue(torch.equal(past_before[key], model.past_model_ckpt[key]))
            self.assertEqual(model.sap_history[-1]['status'], SAP_FAILED)
            self.assertFalse(model._taskwise_artifact_directory(dataset).exists())
            self.assertFalse(list(Path(tmp).rglob('manifest.json')))

    def test_failure_after_commit_rolls_back_both_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(tmp)
            dataset = _Dataset()
            model.past_model_ckpt = copy.deepcopy(model.net.state_dict())
            before = copy.deepcopy(model.net.state_dict())
            original_record = model._record_sap_event

            def fail_success_event(**event):
                if event['status'] == SAP_ORACLE_EXECUTED:
                    raise RuntimeError('event failure')
                return original_record(**event)

            model._record_sap_event = fail_success_event
            with self.assertRaisesRegex(RuntimeError, 'event failure'):
                model._run_taskwise_sap(dataset)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.net.state_dict()[key]))
                self.assertTrue(torch.equal(value, model.past_model_ckpt[key]))
            self.assertEqual(model.sap_history[-1]['status'], SAP_FAILED)
            self.assertFalse(model._taskwise_artifact_directory(dataset).exists())

    def test_alpha_mismatch_fails_before_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _model(tmp)
            model.args.sap_oracle_scale = 100.0
            with self.assertRaisesRegex(ValueError, '300'):
                model._run_taskwise_sap(_Dataset())
