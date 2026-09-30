"""Task0 full SAP and Task1+ partial current-task SAP contracts."""

import argparse
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.aer_sap import AerSap
from models.dgc_sap import SAP_ORACLE_EXECUTED
from tests.test_aer_sap import _Dataset, _model
from tests.test_aer_sap_normalized_cosine_ce import _model as _loss_model
from tests.test_aer_sap_task1_pre_sap_resume import _real_reference_model
from tests.test_aer_sap_task0_resume import _ResumeDataset
from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import Logger


def _load(directory, filename):
    return torch.load(directory / filename, weights_only=True)


class AerSapPartialTests(unittest.TestCase):
    def test_parser_default_and_invalid_beta(self):
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        self.assertEqual(parser.parse_args(['--buffer_size', '4']).sap_incremental_beta, 1.0)
        self.assertEqual(parser.parse_args([
            '--buffer_size', '4', '--sap_incremental_beta', '0.2',
        ]).sap_incremental_beta, 0.2)
        with contextlib.redirect_stderr(io.StringIO()):
            for invalid in ('0', '-0.1', '1.01', 'nan', 'invalid'):
                with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                    parser.parse_args([
                        '--buffer_size', '4', '--sap_incremental_beta', invalid,
                    ])

    def test_beta_one_preserves_full_sap_at_task0_and_task1(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _model(directory)
            model.args.sap_incremental_beta = 1.0
            dataset = _Dataset()
            for task_id in (0, 1):
                model._current_task = task_id
                model._n_seen_classes = (task_id + 1) * 10
                model._run_taskwise_sap(dataset)
                artifact = model._taskwise_artifact_directory(dataset)
                before = _load(artifact, 'W_before.pt')
                after = _load(artifact, 'W_after.pt')
                full = _load(artifact, 'W_full_sap.pt')
                matrix = _load(artifact, f'M_task_{task_id}.pt')
                effective = _load(artifact, f'M_effective_task_{task_id}.pt')
                self.assertTrue(torch.equal(after, full))
                self.assertTrue(torch.equal(matrix, effective))
                start, end = dataset.get_offsets(task_id)
                torch.testing.assert_close(after[start:end], before[start:end] @ matrix.T)
                self.assertTrue(torch.equal(after[:start], before[:start]))
                self.assertTrue(torch.equal(after[end:], before[end:]))
                manifest = json.loads((artifact / 'manifest.json').read_text())
                self.assertEqual(manifest['effective_beta'], 1.0)
                self.assertEqual(manifest['sap_incremental_beta'], 1.0)

    def test_task0_full_then_task1_and_task2_partial_are_formal_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _model(directory)
            model.args.sap_incremental_beta = 0.2
            model.args.training_loss = 'ce'
            model.args.cosine_inference = 0
            dataset = _Dataset()
            preceding_post = None
            for task_id in (0, 1, 2):
                model._current_task = task_id
                model._n_seen_classes = (task_id + 1) * 10
                network_before = copy.deepcopy(model.net.state_dict())
                model._run_taskwise_sap(dataset)
                artifact = model._taskwise_artifact_directory(dataset)
                manifest = json.loads((artifact / 'manifest.json').read_text())
                before = _load(artifact, 'W_before.pt')
                after = _load(artifact, 'W_after.pt')
                full = _load(artifact, 'W_full_sap.pt')
                matrix = _load(artifact, f'M_task_{task_id}.pt')
                effective = _load(artifact, f'M_effective_task_{task_id}.pt')
                start, end = dataset.get_offsets(task_id)
                beta = 1.0 if task_id == 0 else 0.2
                self.assertEqual(manifest['effective_beta'], beta)
                self.assertEqual(manifest['sap_incremental_beta'], 0.2)
                self.assertEqual(manifest['projection_scope'], 'current_task_only')
                self.assertEqual(model.sap_history[-1]['effective_beta'], beta)
                self.assertEqual(model.sap_history[-1]['sap_incremental_beta'], 0.2)
                self.assertTrue(torch.equal(before, network_before['classifier.weight']))
                if preceding_post is not None:
                    self.assertTrue(torch.equal(before, preceding_post))
                self.assertTrue(torch.equal(after[:start], before[:start]))
                self.assertTrue(torch.equal(after[end:], before[end:]))
                self.assertTrue(torch.equal(full[:start], before[:start]))
                self.assertTrue(torch.equal(full[end:], before[end:]))
                torch.testing.assert_close(full[start:end], before[start:end] @ matrix.T)
                if task_id == 0:
                    self.assertTrue(torch.equal(effective, matrix))
                    self.assertTrue(torch.equal(after, full))
                else:
                    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype)
                    torch.testing.assert_close(effective, identity + 0.2 * (matrix - identity),
                                               rtol=0, atol=0)
                    expected = before[start:end] + 0.2 * (full[start:end] - before[start:end])
                    torch.testing.assert_close(after[start:end], expected, rtol=0, atol=0)
                    torch.testing.assert_close(after[start:end],
                                               before[start:end] @ effective.T,
                                               rtol=1e-5, atol=1e-5)
                    self.assertFalse(torch.equal(after[start:end], full[start:end]))
                post_state = _load(artifact, 'network_post_sap.pt')
                self.assertTrue(torch.equal(post_state['classifier.weight'], after))
                for key, value in network_before.items():
                    if key != 'classifier.weight':
                        self.assertTrue(torch.equal(model.net.state_dict()[key], value), key)
                        self.assertTrue(torch.equal(post_state[key], value), key)
                    self.assertTrue(torch.equal(model.past_model_ckpt[key],
                                                model.net.state_dict()[key]), key)
                features = _load(artifact, 'test_post_sap_raw_features.pt')
                logits = _load(artifact, 'test_post_sap_logits.pt')
                expected_logits = features @ after.T + network_before['classifier.bias']
                torch.testing.assert_close(logits, expected_logits)
                self.assertEqual(set(json.loads((artifact / 'accuracy.json').read_text())),
                                 {'pre_sap', 'post_sap'})
                if task_id == 1:
                    self.assertTrue(torch.equal(
                        _load(artifact, 'task1_four_settings_task1_only_W.pt'), full,
                    ))
                    self.assertFalse(torch.equal(
                        _load(artifact, 'task1_four_settings_task1_only_W.pt'), after,
                    ))
                    self.assertTrue(torch.equal(model.net.classifier.weight, after))
                preceding_post = after

    def test_linear_ce_training_scoring_and_inference_are_unchanged(self):
        model, calls = _loss_model('ce')
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        labels = torch.tensor([0, 1])
        expected = model.net(inputs).detach().clone()
        model.observe(inputs, labels, inputs, epoch=0)
        self.assertTrue(torch.equal(calls[0][0], expected))
        self.assertTrue(torch.equal(calls[1][0], expected))
        model.eval()
        torch.testing.assert_close(model.forward(inputs), model.net(inputs), rtol=0, atol=0)

    def test_task1_pre_sap_resume_applies_partial_once_before_task2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / 'checkpoints'
            checkpoint_dir.mkdir()
            dataset = _ResumeDataset()
            dataset.c_task = -1
            dataset.get_data_loaders()
            live = _real_reference_model(str(root / 'live'), dataset)
            live.args.sap_incremental_beta = 0.2
            live.meta_begin_task(dataset)
            for sample_id, label in ((0, 0), (1, 1)):
                live.buffer.add_data(
                    examples=torch.full((1, 3, 32, 32), (sample_id + 1) / 255),
                    labels=torch.tensor([label]), true_labels=torch.tensor([label]),
                    task_labels=torch.tensor([0]), sample_ids=torch.tensor([sample_id]),
                    sample_selection_scores=torch.tensor([(sample_id + 1) / 10]),
                )
            live.meta_end_task(dataset)
            dataset.get_data_loaders()
            live.meta_begin_task(dataset)
            live.seen_so_far = torch.arange(20)
            with torch.no_grad():
                live.net.backbone.weight[0, 1] += 0.25
            live.save_model_checkpoint()
            logger = Logger(live.args, dataset.SETTING, dataset.NAME, live.NAME)
            logger.log((50.0, 50.0))
            logger.log_fullacc(([50.0], [50.0]))
            live._task1_pre_sap_results = [[[50.0]], [[50.0]], logger.dump()]
            with patch('models.aer_sap.get_checkpoint_path', return_value=str(checkpoint_dir)):
                live.meta_end_task(dataset)
            direct_artifact = Path(live.sap_history[-1]['artifact_output_directory'])
            checkpoint = checkpoint_dir / f'{live.args.ckpt_name}_1_pre_sap.pt'
            self.assertTrue(checkpoint.exists())
            direct_before = _load(direct_artifact, 'W_before.pt')
            direct_after = _load(direct_artifact, 'W_after.pt')
            direct_full = _load(direct_artifact, 'W_full_sap.pt')
            torch.testing.assert_close(direct_after[10:20],
                                       direct_before[10:20]
                                       + 0.2 * (direct_full[10:20] - direct_before[10:20]),
                                       rtol=0, atol=0)

            restored_dataset = _ResumeDataset()
            restored_dataset.c_task = -1
            resumed = _real_reference_model(
                str(root / 'live'), restored_dataset,
                loadcheck=str(checkpoint), start_from=2,
            )
            resumed.args.sap_incremental_beta = 0.2
            with patch.object(resumed, '_run_taskwise_sap',
                              side_effect=AssertionError('SAP repeated during reconstruction')):
                for _ in range(2):
                    restored_dataset.get_data_loaders()
                    resumed.meta_begin_task(restored_dataset)
                    resumed.meta_end_task(restored_dataset)
            resumed, _ = mammoth_load_checkpoint(
                str(checkpoint), resumed, args=resumed.args,
            )
            resumed.args.results_path = str(root / 'resumed')
            self.assertTrue(resumed._pending_task1_sap)
            self.assertEqual(resumed.current_task, 2)
            self.assertTrue(resumed.resume_pending_task1_sap(restored_dataset))
            self.assertFalse(resumed.resume_pending_task1_sap(restored_dataset))
            self.assertEqual(resumed.current_task, 2)
            resumed_artifact = Path(resumed.sap_history[-1]['artifact_output_directory'])
            self.assertEqual(resumed.sap_history[-1]['effective_beta'], 0.2)
            for filename in ('W_before.pt', 'W_after.pt', 'W_full_sap.pt',
                             'M_task_1.pt', 'M_effective_task_1.pt'):
                torch.testing.assert_close(_load(direct_artifact, filename),
                                           _load(resumed_artifact, filename),
                                           rtol=1e-6, atol=1e-6)
            for key, value in live.net.state_dict().items():
                torch.testing.assert_close(resumed.net.state_dict()[key], value,
                                           rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(resumed.past_model_ckpt[key], value,
                                           rtol=1e-6, atol=1e-6)
            self.assertEqual(
                [event['task_id'] for event in resumed.sap_history
                 if event['status'] == SAP_ORACLE_EXECUTED],
                [0, 1],
            )


if __name__ == '__main__':
    unittest.main()
