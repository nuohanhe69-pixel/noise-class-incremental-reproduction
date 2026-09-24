"""Task0 Post-SAP task-boundary checkpoint resumes at Task1."""

import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.aer_sap import AerSap
from models.dgc_sap import SAP_ORACLE_EXECUTED
from tests.test_aer_sap import _Dataset, _model
from tests.test_sap_oracle_smoke import _FakeTrainDataset
from utils.buffer import Buffer
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint


class _ResumeDataset(_Dataset):
    def __init__(self):
        super().__init__()
        self.c_task = 0
        true_labels = torch.tensor([10, 10, 11, 11, 10, 11])
        observed = true_labels.clone()
        observed[0] = 11
        self.train_loader = SimpleNamespace(dataset=_FakeTrainDataset(
            len(true_labels), 2, true_labels, observed,
        ))

    def get_data_loaders(self):
        self.c_task += 1
        if self.c_task == 0:
            labels = torch.tensor([0, 0, 1, 1])
        else:
            labels = torch.tensor([10, 10, 11, 11, 10, 11])
        self.train_loader = SimpleNamespace(dataset=_FakeTrainDataset(
            len(labels), 2, labels, labels.clone(),
        ))
        return self.train_loader, self.test_loaders[self.c_task]

    def get_offsets(self, task_id=None):
        task_id = self.c_task if task_id is None else task_id
        return task_id * 10, (task_id + 1) * 10


def _configure_model(directory, dataset, *, loadcheck=None, start_from=None):
    model = _model(directory)
    model.dataset = dataset
    model._cpt = 10
    model.N_CLASSES = 100
    model.num_classes = 100
    model.seen_so_far = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) if loadcheck is None else torch.empty(0, dtype=torch.long)
    model.loss_trace_recorder = None
    model.args = Namespace(**{
        **vars(model.args),
        'buffer_size': 3, 'buffer_fitting_epochs': 0,
        'savecheck': 'task', 'save_checkpoint_mode': 'safe',
        'ckpt_name': 'task0-post-sap', 'joint': False,
        'loadcheck': loadcheck, 'start_from': start_from,
        'inference_only': False, 'force_compat': False,
        'distributed': 'no', 'model': 'aer-sap',
        'optimizer': 'sgd', 'lr': 0.0, 'optim_wd': 0.0,
        'optim_mom': 0.0, 'optim_nesterov': False,
    })
    model.opt = torch.optim.SGD(model.net.parameters(), lr=0.0)
    model.buffer = Buffer(3, device='cpu', dataset=dataset,
                          sample_selection_strategy='abs')
    return model


def _assert_same_buffer(test, left, right):
    test.assertEqual(left.num_seen_examples, right.num_seen_examples)
    for field in ('examples', 'labels', 'true_labels', 'task_labels', 'sample_ids'):
        test.assertTrue(torch.equal(getattr(left, field), getattr(right, field)), field)
    test.assertTrue(torch.equal(
        left.sample_selection_fn.importance_scores,
        right.sample_selection_fn.importance_scores,
    ))


class AerSapTask0ResumeTests(unittest.TestCase):
    def test_post_sap_checkpoint_round_trip_and_task1_start(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _ResumeDataset()
            live = _configure_model(directory, dataset)
            # Retained Task0 samples, including one dirty label, plus an ABS
            # replacement. The checkpoint must retain the selector state too.
            torch.manual_seed(0)
            for sample_id, observed, true in (
                (20, 0, 0), (21, 1, 1), (22, 0, 1), (23, 1, 1),
            ):
                live.buffer.add_data(
                    examples=torch.full((1, 3, 32, 32), sample_id / 255),
                    labels=torch.tensor([observed]),
                    true_labels=torch.tensor([true]),
                    task_labels=torch.tensor([0]),
                    sample_ids=torch.tensor([sample_id]),
                    sample_selection_scores=torch.tensor([sample_id / 100]),
                )
            self.assertEqual(live.buffer.num_seen_examples, 4)
            live.meta_end_task(dataset)  # real AER end_task + first SAP + task advance
            self.assertEqual(live.current_task, 1)
            self.assertEqual([event['task_id'] for event in live.sap_history
                              if event['status'] == SAP_ORACLE_EXECUTED], [0])
            artifact = Path(live.sap_history[-1]['artifact_output_directory'])
            self.assertTrue(json.loads((artifact / 'manifest.json').read_text())['valid'])
            before_save_net = copy.deepcopy(live.net.state_dict())
            before_save_past = copy.deepcopy(live.past_model_ckpt)
            self.assertTrue(all(torch.equal(before_save_net[key], before_save_past[key])
                                for key in before_save_net))

            stem = str(Path(directory) / 'task0-post-sap')
            save_mammoth_checkpoint(0, 10, live.args, live,
                                    optimizer_st=live.opt.state_dict(),
                                    checkpoint_name=stem)
            checkpoint_path = stem + '.pt'
            payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.assertEqual(payload['sap_state']['next_task'], 1)
            self.assertIn('past_model_ckpt', payload['sap_state'])
            self.assertIn('seen_so_far', payload['sap_state'])
            self.assertIn('buffer', payload)
            self.assertTrue(torch.equal(
                payload['sap_state']['past_model_ckpt']['classifier.weight'],
                payload['model']['net.classifier.weight'],
            ))

            # Simulate a fresh process: fresh network, Buffer, dataset and model.
            restored_dataset = _ResumeDataset()
            resumed = _configure_model(directory, restored_dataset,
                                       loadcheck=checkpoint_path, start_from=1)
            with patch.object(resumed, '_run_taskwise_sap',
                              side_effect=AssertionError('Task0 SAP repeated')):
                restored_dataset.c_task = -1
                restored_dataset.get_data_loaders()
                resumed.meta_begin_task(restored_dataset)
                resumed.meta_end_task(restored_dataset)  # checkpoint reconstruction
            self.assertEqual(resumed.current_task, 1)
            self.assertFalse(hasattr(resumed, 'past_model_ckpt'))
            resumed, _ = mammoth_load_checkpoint(
                checkpoint_path, resumed, args=resumed.args,
            )
            self.assertEqual(resumed.current_task, 1)
            self.assertEqual([event['task_id'] for event in resumed.sap_history
                              if event['status'] == SAP_ORACLE_EXECUTED], [0])
            for key, value in before_save_net.items():
                self.assertTrue(torch.equal(resumed.net.state_dict()[key], value), key)
                self.assertTrue(torch.equal(resumed.past_model_ckpt[key], value), key)
            self.assertTrue(torch.equal(resumed.seen_so_far, live.seen_so_far))
            _assert_same_buffer(self, live.buffer, resumed.buffer)

            # Compare the two paths at the Task1 begin boundary. No Task1 SAP
            # or SGD step is needed to establish the legal starting state.
            dataset.get_data_loaders()
            restored_dataset.get_data_loaders()
            live.meta_begin_task(dataset)
            resumed.meta_begin_task(restored_dataset)
            self.assertEqual(live.current_task, resumed.current_task)
            self.assertEqual(live.n_seen_classes, resumed.n_seen_classes)
            self.assertEqual(live.n_past_classes, resumed.n_past_classes)
            self.assertTrue(torch.equal(live.seen_so_far, resumed.seen_so_far))
            _assert_same_buffer(self, live.buffer, resumed.buffer)
            for key, value in live.net.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.net.state_dict()[key]), key)
                self.assertTrue(torch.equal(live.past_model_ckpt[key],
                                            resumed.past_model_ckpt[key]), key)

            direct = AerSap._build_oracle_reference_batches(
                live, dataset, return_task_ids=True, return_evidence=True,
            )
            recovered = AerSap._build_oracle_reference_batches(
                resumed, restored_dataset, return_task_ids=True,
                return_evidence=True,
            )
            for field in range(3):  # images, true labels, task IDs
                self.assertTrue(torch.equal(direct[field], recovered[field]))
            self.assertEqual(direct[3], recovered[3])
            old_start = direct[3]['reference_new_count']
            for key in ('sample_ids', 'observed_labels', 'true_labels',
                        'source_task_ids', 'is_old'):
                self.assertTrue(torch.equal(direct[4][key][old_start:],
                                            recovered[4][key][old_start:]), key)

            with torch.no_grad():
                resumed.net.classifier.weight.add_(1)
            resumed.begin_epoch(1, restored_dataset)
            for key, value in before_save_net.items():
                self.assertTrue(torch.equal(resumed.net.state_dict()[key], value), key)

    def test_missing_sap_state_or_wrong_next_task_rejects_training_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _ResumeDataset()
            model = _configure_model(directory, dataset)
            model.meta_end_task(dataset)
            stem = str(Path(directory) / 'task0-post-sap')
            save_mammoth_checkpoint(0, 10, model.args, model,
                                    checkpoint_name=stem)
            path = stem + '.pt'
            raw = torch.load(path, map_location='cpu', weights_only=False)
            raw.pop('sap_state')
            legacy = str(Path(directory) / 'missing-sap-state.pt')
            torch.save(raw, legacy)
            fresh = _configure_model(directory, _ResumeDataset(),
                                     loadcheck=legacy, start_from=1)
            fresh._current_task = 1
            with self.assertRaisesRegex(ValueError, 'requires SAP checkpoint state'):
                mammoth_load_checkpoint(legacy, fresh, args=fresh.args)
            wrong_task = _configure_model(directory, _ResumeDataset(),
                                          loadcheck=path, start_from=None)
            with self.assertRaisesRegex(ValueError, 'requires start_from=1'):
                mammoth_load_checkpoint(path, wrong_task, args=wrong_task.args)
            corrupted = torch.load(path, map_location='cpu', weights_only=False)
            corrupted['sap_state']['past_model_ckpt']['classifier.weight'][0, 0] += 1
            corrupt_path = str(Path(directory) / 'mismatched-aer.pt')
            torch.save(corrupted, corrupt_path)
            mismatched = _configure_model(directory, _ResumeDataset(),
                                          loadcheck=corrupt_path, start_from=1)
            mismatched._current_task = 1
            with self.assertRaisesRegex(ValueError, 'differs from post-SAP network'):
                mammoth_load_checkpoint(corrupt_path, mismatched, args=mismatched.args)
