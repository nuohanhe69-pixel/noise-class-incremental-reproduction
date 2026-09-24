"""Task1 AER boundary checkpoint resumes exactly one pending SAP operation."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.aer_sap import AerSap
from models.dgc_sap import SAP_ORACLE_EXECUTED
from tests.test_aer_sap_task0_resume import (
    _ResumeDataset, _assert_same_buffer, _configure_model,
)
from tests.test_sap_oracle_smoke import _FakeTrainDataset
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint
from utils.loggers import Logger
from utils.training import train


def _real_reference_model(directory, dataset, *, loadcheck=None, start_from=None):
    model = _configure_model(directory, dataset, loadcheck=loadcheck,
                             start_from=start_from)
    model._build_oracle_reference_batches = (
        AerSap._build_oracle_reference_batches.__get__(model, AerSap)
    )
    model._normalized_batches = lambda images, labels: iter([
        (images.reshape(len(images), -1)[:, :4].float(), labels),
    ])
    return model


def _successful_tasks(model):
    return [event['task_id'] for event in model.sap_history
            if event['status'] == SAP_ORACLE_EXECUTED]


class _TrainingLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 1

    def __iter__(self):
        return iter((None,))


class _TrainingDataset(_ResumeDataset):
    def __init__(self):
        super().__init__()
        self.c_task = -1

    def get_data_loaders(self):
        super().get_data_loaders()
        self.train_loader = _TrainingLoader(self.train_loader.dataset)
        return self.train_loader, self.test_loaders[self.c_task]

    def evaluate(self, model, _dataset):
        seen_end = (self.c_task + 1) * 10
        class_accs, task_accs = [], []
        with torch.no_grad():
            for task_id in range(self.c_task + 1):
                inputs, labels = self.test_loaders[task_id].batches[0]
                logits = model.net(inputs)
                class_accs.append(float((logits[:, :seen_end].argmax(1) == labels).float().mean() * 100))
                start, end = self.get_offsets(task_id)
                task_accs.append(float((logits[:, start:end].argmax(1) + start == labels).float().mean() * 100))
        return class_accs, task_accs

    def log(self, args, logger, accs, task, setting, **_kwargs):
        logger.log((sum(accs[0]) / len(accs[0]), sum(accs[1]) / len(accs[1])))
        logger.log_fullacc(accs)
        return accs


def _training_model(directory, dataset, *, loadcheck=None, start_from=None, ckpt_name='direct'):
    model = _real_reference_model(directory, dataset, loadcheck=loadcheck,
                                  start_from=start_from)
    model.args.ckpt_name = ckpt_name
    model.args.stop_after = 2
    model.args.nowand = True
    model.args.disable_log = False
    model.args.eval_future = False
    model.args.enable_other_metrics = False
    model.args.n_epochs = 1
    model.args.fitting_mode = 'epochs'
    model.args.early_stopping_patience = 1
    model.args.eval_epochs = None
    model.args.non_verbose = True
    model.args.code_optimization = 0
    model.args.device = 'cpu'
    model.args.validation = None
    model.args.lr_scheduler = None
    return model


class AerSapTask1PreSapResumeTests(unittest.TestCase):
    def test_train_restores_task1_history_and_post_sap_checkpoint_before_task2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / 'checkpoints'
            checkpoint_dir.mkdir()
            artifacts = str(root / 'artifacts')
            source_dataset = _TrainingDataset()
            source_dataset.c_task = -1
            source_dataset.get_data_loaders()
            source = _training_model(artifacts, source_dataset)
            source.meta_begin_task(source_dataset)
            for sample_id, label in ((0, 0), (1, 1)):
                source.buffer.add_data(
                    examples=torch.full((1, 3, 32, 32), (sample_id + 1) / 255),
                    labels=torch.tensor([label]), true_labels=torch.tensor([label]),
                    task_labels=torch.tensor([0]), sample_ids=torch.tensor([sample_id]),
                    sample_selection_scores=torch.tensor([(sample_id + 1) / 10]),
                )
            source.meta_end_task(source_dataset)
            history_logger = Logger(source.args, source_dataset.SETTING,
                                    source_dataset.NAME, source.NAME)
            history_logger.log((50.0, 50.0))
            history_logger.log_fullacc(([50.0], [50.0]))
            task0_checkpoint = root / 'task0-post-sap'
            save_mammoth_checkpoint(
                0, 2, source.args, source,
                results=[[[50.0]], [[50.0]], history_logger.dump()],
                checkpoint_name=str(task0_checkpoint),
            )

            def one_epoch(model, *_args, **_kwargs):
                self.assertEqual(model.current_task, 1)  # Task1 is trained only in the direct path
                with torch.no_grad():
                    model.net.backbone.weight[0, 1] += 0.25
                model.seen_so_far = torch.arange(20)

            direct_dataset = _TrainingDataset()
            direct = _training_model(artifacts, direct_dataset,
                                     loadcheck=str(task0_checkpoint) + '.pt', start_from=1)
            with patch('models.aer_sap.get_checkpoint_path', return_value=str(checkpoint_dir)), \
                 patch('utils.checkpoints.get_checkpoint_path', return_value=str(checkpoint_dir)), \
                 patch('utils.training.MammothDatasetWrapper', _FakeTrainDataset), \
                 patch('utils.training.train_single_epoch', side_effect=one_epoch) as train_epoch, \
                 patch.object(Logger, 'write'):
                train(direct, direct_dataset, args=direct.args)
            self.assertEqual(train_epoch.call_count, 1)
            self.assertEqual(_successful_tasks(direct), [0, 1])
            pre_checkpoint = checkpoint_dir / 'direct_1_pre_sap.pt'
            direct_post = torch.load(checkpoint_dir / 'direct_1.pt',
                                     map_location='cpu', weights_only=False)
            pre = torch.load(pre_checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(pre['results'][0], [[50.0]])
            self.assertEqual(pre['results'][1], [[50.0]])
            self.assertEqual(pre['results'][2], history_logger.dump())
            self.assertEqual(len(direct_post['results'][0]), 2)
            self.assertEqual(len(direct_post['results'][2]['accs']), 2)
            direct_net = copy.deepcopy(direct.net.state_dict())
            shutil.rmtree(Path(direct.sap_history[-1]['artifact_output_directory']))

            resumed_dataset = _TrainingDataset()
            resumed = _training_model(artifacts, resumed_dataset,
                                      loadcheck=str(pre_checkpoint), start_from=2,
                                      ckpt_name='resumed')
            with patch('models.aer_sap.get_checkpoint_path', return_value=str(checkpoint_dir)), \
                 patch('utils.checkpoints.get_checkpoint_path', return_value=str(checkpoint_dir)), \
                 patch('utils.training.MammothDatasetWrapper', _FakeTrainDataset), \
                 patch('utils.training.train_single_epoch',
                       side_effect=AssertionError('Task1 was retrained')), \
                 patch.object(Logger, 'write'):
                train(resumed, resumed_dataset, args=resumed.args)
            resumed_post = torch.load(checkpoint_dir / 'resumed_1.pt',
                                      map_location='cpu', weights_only=False)
            self.assertEqual(direct_post['results'], resumed_post['results'])
            self.assertEqual(_successful_tasks(resumed), [0, 1])
            self.assertEqual(resumed.current_task, 2)
            for key, value in direct_net.items():
                torch.testing.assert_close(value, resumed.net.state_dict()[key],
                                           rtol=0, atol=0)
                torch.testing.assert_close(direct_post['model'][f'net.{key}'],
                                           resumed_post['model'][f'net.{key}'],
                                           rtol=0, atol=0)

    def test_pre_sap_round_trip_matches_uninterrupted_task1_sap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / 'checkpoints'
            checkpoint_dir.mkdir()
            dataset = _ResumeDataset()
            dataset.c_task = -1
            dataset.get_data_loaders()
            live = _real_reference_model(str(root / 'live'), dataset)
            live.meta_begin_task(dataset)
            for sample_id, observed, true in ((0, 0, 0), (1, 1, 1), (2, 0, 1)):
                live.buffer.add_data(
                    examples=torch.full((1, 3, 32, 32), (sample_id + 1) / 255),
                    labels=torch.tensor([observed]),
                    true_labels=torch.tensor([true]),
                    task_labels=torch.tensor([0]),
                    sample_ids=torch.tensor([sample_id]),
                    sample_selection_scores=torch.tensor([(sample_id + 1) / 10]),
                )
            live.meta_end_task(dataset)
            self.assertEqual(_successful_tasks(live), [0])
            dataset.get_data_loaders()
            live.meta_begin_task(dataset)
            live.seen_so_far = torch.arange(20)
            with torch.no_grad():
                live.net.backbone.weight[0, 1] += 0.25
            live.save_model_checkpoint()  # final AER fitting epoch's checkpoint
            pre_net = copy.deepcopy(live.net.state_dict())
            direct_reference = live._build_oracle_reference_batches(
                dataset, return_task_ids=True, return_evidence=True,
            )
            logger = Logger(live.args, dataset.SETTING, dataset.NAME, live.NAME)
            logger.log((50.0, 50.0))
            logger.log_fullacc(([50.0], [50.0]))
            live._task1_pre_sap_results = [[[50.0]], [[50.0]], logger.dump()]

            with patch('models.aer_sap.get_checkpoint_path',
                       return_value=str(checkpoint_dir)):
                live.meta_end_task(dataset)
            self.assertEqual(_successful_tasks(live), [0, 1])
            direct_artifact = Path(live.sap_history[-1]['artifact_output_directory'])
            checkpoint = checkpoint_dir / f'{live.args.ckpt_name}_1_pre_sap.pt'
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(payload['sap_state']['phase'], 'task1_pre_sap')
            self.assertEqual(payload['sap_state']['next_task'], 2)
            self.assertEqual(payload['results'], live._task1_pre_sap_results)
            self.assertEqual([event['task_id'] for event in payload['sap_state']['history']
                              if event['status'] == SAP_ORACLE_EXECUTED], [0])
            for key, value in pre_net.items():
                torch.testing.assert_close(payload['model'][f'net.{key}'], value, rtol=0, atol=0)
            self.assertTrue(torch.equal(
                payload['model']['net.classifier.weight'],
                torch.load(direct_artifact / 'W_before.pt', weights_only=True),
            ))

            restored_dataset = _ResumeDataset()
            restored_dataset.c_task = -1
            resumed = _real_reference_model(
                str(root / 'live'), restored_dataset,
                loadcheck=str(checkpoint), start_from=2,
            )
            with patch.object(resumed, '_run_taskwise_sap',
                              side_effect=AssertionError('SAP repeated during reconstruction')), \
                 patch('models.aer_sap.save_mammoth_checkpoint',
                       side_effect=AssertionError('checkpoint rewritten during reconstruction')):
                for _ in range(2):
                    restored_dataset.get_data_loaders()
                    resumed.meta_begin_task(restored_dataset)
                    resumed.meta_end_task(restored_dataset)
            self.assertEqual(resumed.current_task, 2)
            resumed, _ = mammoth_load_checkpoint(
                str(checkpoint), resumed, args=resumed.args,
            )
            self.assertTrue(resumed._pending_task1_sap)
            self.assertEqual(_successful_tasks(resumed), [0])
            self.assertEqual(resumed.n_seen_classes, 20)
            self.assertEqual(resumed.n_past_classes, 10)
            self.assertTrue(torch.equal(resumed.seen_so_far, live.seen_so_far))
            for key, value in pre_net.items():
                torch.testing.assert_close(resumed.net.state_dict()[key], value, rtol=0, atol=0)
                torch.testing.assert_close(resumed.past_model_ckpt[key], value, rtol=0, atol=0)
            _assert_same_buffer(self, live.buffer, resumed.buffer)

            resumed._current_task = 1
            restored_reference = resumed._build_oracle_reference_batches(
                restored_dataset, return_task_ids=True, return_evidence=True,
            )
            resumed._current_task = 2
            for index in range(3):
                torch.testing.assert_close(direct_reference[index], restored_reference[index],
                                           rtol=0, atol=0)
            self.assertEqual(direct_reference[3], restored_reference[3])
            for field in ('sample_ids', 'true_labels', 'observed_labels',
                          'source_task_ids', 'is_old', 'reference_order'):
                torch.testing.assert_close(direct_reference[4][field],
                                           restored_reference[4][field], rtol=0, atol=0)

            resumed.args.results_path = str(root / 'resumed')
            resumed.resume_pending_task1_sap(restored_dataset)
            resumed.resume_pending_task1_sap(restored_dataset)
            self.assertFalse(resumed._pending_task1_sap)
            self.assertEqual(resumed.current_task, 2)
            self.assertEqual(_successful_tasks(resumed), [0, 1])
            resumed_artifact = Path(resumed.sap_history[-1]['artifact_output_directory'])
            direct_manifest = json.loads((direct_artifact / 'manifest.json').read_text())
            resumed_manifest = json.loads((resumed_artifact / 'manifest.json').read_text())
            self.assertEqual(direct_manifest, resumed_manifest)
            self.assertTrue(direct_manifest['valid'] and direct_manifest['artifact_complete'])
            self.assertEqual(direct_manifest['seen_tasks'], [0, 1])
            for filename in direct_manifest['files']:
                direct_file = direct_artifact / filename
                resumed_file = resumed_artifact / filename
                if filename.endswith('.json'):
                    self.assertEqual(json.loads(direct_file.read_text()),
                                     json.loads(resumed_file.read_text()), filename)
                elif filename.startswith('eigenvectors_'):
                    continue  # eigenspace signs are not part of the equivalence contract
                else:
                    left = torch.load(direct_file, map_location='cpu', weights_only=False)
                    right = torch.load(resumed_file, map_location='cpu', weights_only=False)
                    if isinstance(left, dict):
                        self.assertEqual(left.keys(), right.keys(), filename)
                        for key in left:
                            torch.testing.assert_close(left[key], right[key],
                                                       rtol=1e-6, atol=1e-6,
                                                       msg=f'{filename}:{key}')
                    else:
                        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6,
                                                   msg=filename)
            for key, value in live.net.state_dict().items():
                torch.testing.assert_close(value, resumed.net.state_dict()[key],
                                           rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(value, resumed.past_model_ckpt[key],
                                           rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
