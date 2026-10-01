"""Contracts for all-seen task-wise, direction-only AER/ABS SAP."""

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

from models.aer_sap import AerSap
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
        for task_id in range(2, self.N_TASKS):
            self.test_loaders.append(_Loader(
                torch.tensor([[2., 0., 0., 0.], [0., 2., 0., 0.]]),
                torch.tensor([task_id * 10, task_id * 10 + 1]),
                [100 * (task_id + 1), 100 * (task_id + 1) + 1],
            ))

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
        # Two oracle-clean buffer samples per historical task. The current
        # reference has the same total count, balanced across its two classes.
        current_images = torch.tensor([
            [0., 0., 255., 0.], [0., 0., 0., 255.],
        ]).repeat(task_id, 1)
        old_images = torch.tensor([
            [2., 0., 0., 0.], [0., 2., 0., 0.],
        ]).repeat(task_id, 1)
        images = torch.cat([current_images, old_images])
        current_labels = torch.tensor([task_id * 10, task_id * 10 + 1]).repeat(task_id)
        old_labels = torch.cat([
            torch.tensor([old_task * 10, old_task * 10 + 1])
            for old_task in range(task_id)
        ])
        labels = torch.cat([current_labels, old_labels])
        ids = torch.cat([current_labels, old_labels])
        tasks = torch.cat([
            torch.full((2 * task_id,), task_id, dtype=torch.long),
            torch.arange(task_id).repeat_interleave(2),
        ])
        new_count = 2 * task_id
    counts = {
        'current_candidate': new_count, 'current_eligible': new_count,
        'current_selected': new_count,
        'old_candidate': len(labels) - new_count,
        'old_eligible': len(labels) - new_count,
        'old_selected': len(labels) - new_count,
        'new_candidate': new_count,
        'new_eligible': new_count, 'new_selected': new_count,
        'selected_per_class': {
            int(label): int((labels == label).sum()) for label in labels.unique()
        },
        'sampling_seed': task_id,
    }
    evidence = {
        'sample_ids': ids, 'true_labels': labels.clone(),
        'observed_labels': labels.clone(), 'source_task_ids': tasks.clone(),
        'is_old': torch.arange(len(labels)) >= new_count,
        'reference_order': torch.arange(len(labels)), 'counts': counts,
    }
    return images, labels, tasks, {
        'reference_new_count': new_count,
        'reference_old_count': len(labels) - new_count,
    }, evidence


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
        self.assertEqual([task for kind, task in order if kind == 'sap'], list(range(10)))
        self.assertEqual(order[:4], [('aer', 0), ('sap', 0), ('aer', 1), ('sap', 1)])
        self.assertEqual(len(order), 20)

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
            self.assertTrue((second / 'M_task_0.pt').exists())
            self.assertTrue((second / 'M_task_1.pt').exists())
            self.assertEqual(first_matrix.shape,
                             torch.load(second / 'M_task_1.pt', weights_only=True).shape)

            before = torch.load(second / 'W_before.pt', weights_only=True)
            full = torch.load(second / 'W_full_sap.pt', weights_only=True)
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
                torch.testing.assert_close(full[start:end], before[start:end] @ matrix.T)
            torch.testing.assert_close(after[:20].norm(dim=1), before[:20].norm(dim=1))
            torch.testing.assert_close(
                torch.nn.functional.normalize(after[:20], dim=1),
                torch.nn.functional.normalize(full[:20], dim=1),
            )
            self.assertTrue(torch.equal(
                torch.load(second / 'bias_before.pt', weights_only=True),
                torch.load(second / 'bias_after.pt', weights_only=True),
            ))
            post_state = torch.load(second / 'network_post_sap.pt', weights_only=True)
            self.assertTrue(torch.equal(post_state['classifier.weight'], after))
            for key, value in pre_state.items():
                if key != 'classifier.weight':
                    self.assertTrue(torch.equal(value, post_state[key]))
                self.assertTrue(torch.equal(model.past_model_ckpt[key], model.net.state_dict()[key]))
                self.assertTrue(torch.equal(model.past_model_ckpt[key], post_state[key]))

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

    def test_each_boundary_projects_each_seen_task_and_preserves_row_lengths(self):
        for task_id in range(10):
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as tmp:
                model = _model(tmp)
                dataset = _Dataset()
                model._current_task = task_id
                model._n_seen_classes = (task_id + 1) * 10
                before = copy.deepcopy(model.net.state_dict())
                model._run_taskwise_sap(dataset)
                artifact = model._taskwise_artifact_directory(dataset)
                manifest = json.loads((artifact / 'manifest.json').read_text())
                self.assertEqual(manifest['boundary_task_id'], task_id)
                self.assertEqual(manifest['projection_scope'],
                                 'taskwise_seen_tasks_direction_only')
                self.assertEqual(manifest['seen_tasks'], list(range(task_id + 1)))
                self.assertEqual(manifest['sap_alpha'], 300.0)
                self.assertTrue(manifest['valid'] and manifest['artifact_complete'])
                self.assertEqual(set(manifest['files']),
                                 {path.name for path in artifact.iterdir()} - {'manifest.json'})
                for seen_task in range(task_id + 1):
                    self.assertTrue((artifact / f'G_task_{seen_task}.pt').exists())
                    self.assertTrue((artifact / f'M_task_{seen_task}.pt').exists())
                for future_task in range(task_id + 1, dataset.N_TASKS):
                    self.assertFalse((artifact / f'G_task_{future_task}.pt').exists())
                    self.assertFalse((artifact / f'M_task_{future_task}.pt').exists())
                x = torch.load(artifact / 'X_l2.pt', weights_only=True)
                x_raw = torch.load(artifact / 'X_raw.pt', weights_only=True)
                source_tasks = torch.load(artifact / 'reference_source_task_ids.pt',
                                          weights_only=True)
                is_old = torch.load(artifact / 'reference_is_old.pt', weights_only=True)
                self.assertTrue(torch.equal(source_tasks == task_id, ~is_old))
                if task_id:
                    torch.testing.assert_close(
                        x_raw[:2], torch.eye(4)[2:4] @ before['backbone.weight'].T,
                    )
                weight_before = torch.load(artifact / 'W_before.pt', weights_only=True)
                weight_full = torch.load(artifact / 'W_full_sap.pt', weights_only=True)
                weight_after = torch.load(artifact / 'W_after.pt', weights_only=True)
                for seen_task in range(task_id + 1):
                    mask = source_tasks == seen_task
                    task_x = x[mask]
                    torch.testing.assert_close(
                        torch.load(artifact / f'X_raw_task_{seen_task}.pt', weights_only=True),
                        x_raw[mask],
                    )
                    torch.testing.assert_close(
                        torch.load(artifact / f'X_l2_task_{seen_task}.pt', weights_only=True),
                        task_x,
                    )
                    gram = torch.load(artifact / f'G_task_{seen_task}.pt', weights_only=True)
                    matrix = torch.load(artifact / f'M_task_{seen_task}.pt', weights_only=True)
                    eigenvalues = torch.load(artifact / f'eigenvalues_task_{seen_task}.pt',
                                             weights_only=True)
                    energy = torch.load(artifact / f'energy_task_{seen_task}.pt',
                                        weights_only=True)
                    eigenvectors = torch.load(artifact / f'eigenvectors_task_{seen_task}.pt',
                                              weights_only=True)
                    importance = torch.load(artifact / f'importance_task_{seen_task}.pt',
                                            weights_only=True)
                    torch.testing.assert_close(gram, task_x.T @ task_x)
                    torch.testing.assert_close(energy, eigenvalues.clamp_min(0))
                    torch.testing.assert_close(
                        matrix, (eigenvectors * importance) @ eigenvectors.T,
                    )
                    start, end = dataset.get_offsets(seen_task)
                    torch.testing.assert_close(
                        weight_full[start:end], weight_before[start:end] @ matrix.T,
                    )
                    torch.testing.assert_close(
                        weight_after[start:end].norm(dim=1),
                        weight_before[start:end].norm(dim=1),
                    )
                    torch.testing.assert_close(
                        torch.nn.functional.normalize(weight_after[start:end], dim=1),
                        torch.nn.functional.normalize(weight_full[start:end], dim=1),
                    )
                _, end = dataset.get_offsets(task_id)
                self.assertTrue(torch.equal(weight_full[end:], weight_before[end:]))
                self.assertTrue(torch.equal(weight_after[end:], weight_before[end:]))
                self.assertTrue(torch.equal(model.net.classifier.bias,
                                            before['classifier.bias']))
                self.assertTrue(torch.equal(model.net.backbone.weight,
                                            before['backbone.weight']))
                post_state = torch.load(artifact / 'network_post_sap.pt', weights_only=True)
                self.assertTrue(torch.equal(post_state['classifier.weight'], weight_after))
                for key, value in post_state.items():
                    self.assertTrue(torch.equal(model.past_model_ckpt[key], value))
                    self.assertTrue(torch.equal(model.net.state_dict()[key], value))
                self.assertEqual(model.sap_history[-1]['status'], SAP_ORACLE_EXECUTED)
                self.assertEqual(model.sap_history[-1]['sap_alpha'], 300.0)

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
