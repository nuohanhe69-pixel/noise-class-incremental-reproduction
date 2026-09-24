"""Task1 AER boundary checkpoint resumes exactly one pending SAP operation."""

import copy
import json
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
from utils.checkpoints import mammoth_load_checkpoint


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


class AerSapTask1PreSapResumeTests(unittest.TestCase):
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

            with patch('models.aer_sap.get_checkpoint_path',
                       return_value=str(checkpoint_dir)):
                live.meta_end_task(dataset)
            self.assertEqual(_successful_tasks(live), [0, 1])
            direct_artifact = Path(live.sap_history[-1]['artifact_output_directory'])
            checkpoint = checkpoint_dir / f'{live.args.ckpt_name}_1_pre_sap.pt'
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(payload['sap_state']['phase'], 'task1_pre_sap')
            self.assertEqual(payload['sap_state']['next_task'], 2)
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
