"""Full and direction-only writes share the current-task SAP projection."""

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from models.aer_sap import AerSap
from models.dgc_sap import SAP_ORACLE_EXECUTED
from tests.test_aer_sap import _Dataset, _model
from tests.test_aer_sap_cosine_inference import _EvaluationDataset
from tests.test_aer_sap_task0_resume import _ResumeDataset, _configure_model
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint


def _load(directory, filename):
    return torch.load(directory / filename, weights_only=True)


def _set_distinct_rows(model):
    generator = torch.Generator().manual_seed(37)
    weight = torch.randn(model.net.classifier.weight.shape, generator=generator)
    weight *= torch.linspace(0.1, 4.0, len(weight)).unsqueeze(1)
    with torch.no_grad():
        model.net.classifier.weight.copy_(weight)


class AerSapWeightCommitTests(unittest.TestCase):
    def test_parser_defaults_to_full_and_accepts_direction_only(self):
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        self.assertEqual(parser.parse_args(['--buffer_size', '20']).sap_weight_commit,
                         'full')
        for mode in ('full', 'direction_only'):
            with self.subTest(mode=mode):
                args = parser.parse_args(['--buffer_size', '20',
                                          '--sap_weight_commit', mode])
                self.assertEqual(args.sap_weight_commit, mode)
                self.assertEqual(args.sap_oracle_scale, 300.0)

    def test_default_and_explicit_full_preserve_parent_projection_exactly(self):
        for task_id in (0, 1):
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as tmp:
                initial = _model(tmp)
                _set_distinct_rows(initial)
                initial_state = copy.deepcopy(initial.net.state_dict())
                results = []
                for mode in (None, 'full'):
                    model = _model(Path(tmp) / str(mode))
                    model.net.load_state_dict(initial_state)
                    model._current_task = task_id
                    model._n_seen_classes = (task_id + 1) * 10
                    if mode is not None:
                        model.args.sap_weight_commit = mode
                    dataset = _Dataset()
                    model._run_taskwise_sap(dataset)
                    artifact = model._taskwise_artifact_directory(dataset)
                    before = _load(artifact, 'W_before.pt')
                    matrix = _load(artifact, f'M_task_{task_id}.pt')
                    after = _load(artifact, 'W_after.pt')
                    start, end = dataset.get_offsets(task_id)
                    expected = before.clone()
                    expected[start:end] = before[start:end] @ matrix.T
                    torch.testing.assert_close(after, expected, rtol=0, atol=0)
                    self.assertFalse((artifact / 'W_full_sap.pt').exists())
                    manifest = json.loads((artifact / 'manifest.json').read_text())
                    self.assertEqual(manifest['sap_weight_commit'], 'full')
                    self.assertEqual(manifest['projection_scope'], 'current_task_only')
                    self.assertEqual(model.sap_history[-1]['sap_weight_commit'], 'full')
                    for key, value in model.net.state_dict().items():
                        torch.testing.assert_close(model.past_model_ckpt[key], value,
                                                   rtol=0, atol=0)
                    results.append(after)
                torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_direction_only_restores_each_current_row_and_saves_full_projection(self):
        for task_id in (0, 1, 9):
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as tmp:
                model = _model(tmp)
                _set_distinct_rows(model)
                model.args.sap_weight_commit = 'direction_only'
                model._current_task = task_id
                model._n_seen_classes = (task_id + 1) * 10
                before_state = copy.deepcopy(model.net.state_dict())
                dataset = _Dataset()
                model._run_taskwise_sap(dataset)
                artifact = model._taskwise_artifact_directory(dataset)
                before = _load(artifact, 'W_before.pt')
                after = _load(artifact, 'W_after.pt')
                full = _load(artifact, 'W_full_sap.pt')
                matrix = _load(artifact, f'M_task_{task_id}.pt')
                start, end = dataset.get_offsets(task_id)
                expected_full = before.clone()
                expected_full[start:end] = before[start:end] @ matrix.T
                torch.testing.assert_close(full, expected_full, rtol=0, atol=0)
                torch.testing.assert_close(before, before_state['classifier.weight'],
                                           rtol=0, atol=0)
                expected_after = before.clone()
                expected_after[start:end] = (
                    F.normalize(full[start:end], p=2, dim=1)
                    * before[start:end].norm(p=2, dim=1, keepdim=True)
                )
                torch.testing.assert_close(after, expected_after, rtol=0, atol=0)
                torch.testing.assert_close(after[start:end].norm(p=2, dim=1),
                                           before[start:end].norm(p=2, dim=1),
                                           rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(F.normalize(after[start:end], p=2, dim=1),
                                           F.normalize(full[start:end], p=2, dim=1),
                                           rtol=1e-6, atol=1e-6)
                # Distinct restoration factors make a common rescaling fail.
                factors = (before[start:end].norm(p=2, dim=1)
                           / full[start:end].norm(p=2, dim=1))
                self.assertFalse(torch.allclose(factors, factors[0].expand_as(factors)))
                self.assertFalse(torch.equal(after[start:end], full[start:end]))
                self.assertTrue(torch.equal(after[:start], before[:start]))
                self.assertTrue(torch.equal(after[end:], before[end:]))
                post_state = _load(artifact, 'network_post_sap.pt')
                for key, value in model.net.state_dict().items():
                    torch.testing.assert_close(post_state[key], value, rtol=0, atol=0)
                    torch.testing.assert_close(model.past_model_ckpt[key], value,
                                               rtol=0, atol=0)
                    if key != 'classifier.weight':
                        torch.testing.assert_close(value, before_state[key], rtol=0, atol=0)
                torch.testing.assert_close(_load(artifact, 'bias_after.pt'),
                                           before_state['classifier.bias'], rtol=0, atol=0)
                manifest = json.loads((artifact / 'manifest.json').read_text())
                self.assertEqual(manifest['sap_weight_commit'], 'direction_only')
                self.assertEqual(manifest['projection_scope'], 'current_task_only')
                self.assertEqual(manifest['sap_alpha'], 300.0)
                self.assertEqual(manifest['reference_count'], 2 if task_id == 0 else 4)
                self.assertIn('W_full_sap.pt', manifest['files'])
                self.assertEqual(set(manifest['files']),
                                 {path.name for path in artifact.iterdir()} - {'manifest.json'})
                self.assertEqual(model.sap_history[-1]['status'], SAP_ORACLE_EXECUTED)
                for event in model.sap_history:
                    self.assertEqual(event['sap_weight_commit'], 'direction_only')

    def test_cosine_predictions_match_full_for_the_same_boundary_state(self):
        for task_id in (0, 1):
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as tmp:
                full = _model(Path(tmp) / 'full')
                _set_distinct_rows(full)
                direction = _model(Path(tmp) / 'direction')
                direction.net.load_state_dict(full.net.state_dict())
                artifacts = []
                for model, mode in ((full, 'full'), (direction, 'direction_only')):
                    model.args.sap_weight_commit = mode
                    model.args.cosine_inference = 1
                    model.args.eval_future = False
                    model._current_task = task_id
                    model._n_seen_classes = (task_id + 1) * 10
                    dataset = _EvaluationDataset()
                    dataset.c_task = task_id
                    dataset.test_loaders = dataset.test_loaders[:task_id + 1]
                    model._run_taskwise_sap(dataset)
                    model.net.eval()
                    artifacts.append(model._taskwise_artifact_directory(dataset))
                torch.testing.assert_close(_load(artifacts[1], 'W_full_sap.pt'),
                                           _load(artifacts[0], 'W_after.pt'), rtol=0, atol=0)
                for name in ('X_l2.pt', f'M_task_{task_id}.pt'):
                    torch.testing.assert_close(_load(artifacts[0], name),
                                               _load(artifacts[1], name), rtol=0, atol=0)
                for name in ('predictions', 'task_predictions'):
                    self.assertTrue(torch.equal(
                        _load(artifacts[0], f'test_post_sap_{name}.pt'),
                        _load(artifacts[1], f'test_post_sap_{name}.pt'),
                    ))
                inputs = torch.randn(64, 4, generator=torch.Generator().manual_seed(103))
                with torch.no_grad():
                    full_logits, direction_logits = full(inputs), direction(inputs)
                self.assertTrue(torch.equal(full_logits.argmax(1), direction_logits.argmax(1)))
                for seen_task in range(task_id + 1):
                    start, end = dataset.get_offsets(seen_task)
                    self.assertTrue(torch.equal(full_logits[:, start:end].argmax(1),
                                                direction_logits[:, start:end].argmax(1)))
                # The mean-row-norm scale changes while normalized directions agree.
                self.assertFalse(torch.allclose(full_logits, direction_logits))
                seen = full.n_seen_classes
                full_scale = full.net.classifier.weight[:seen].norm(p=2, dim=1).mean()
                direction_scale = direction.net.classifier.weight[:seen].norm(p=2, dim=1).mean()
                torch.testing.assert_close(full_logits / full_scale,
                                           direction_logits / direction_scale,
                                           rtol=1e-6, atol=1e-6)

    def test_direction_only_checkpoint_and_next_task_aer_use_committed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _ResumeDataset()
            live = _configure_model(tmp, dataset)
            _set_distinct_rows(live)
            live.args.sap_weight_commit = 'direction_only'
            live.meta_end_task(dataset)
            self.assertEqual(live.current_task, 1)
            committed = copy.deepcopy(live.net.state_dict())
            artifact = Path(live.sap_history[-1]['artifact_output_directory'])
            self.assertFalse(torch.equal(committed['classifier.weight'],
                                         _load(artifact, 'W_full_sap.pt')))
            stem = str(Path(tmp) / 'direction-only-post-sap')
            save_mammoth_checkpoint(0, 10, live.args, live,
                                    optimizer_st=live.opt.state_dict(), checkpoint_name=stem)
            restored_dataset = _ResumeDataset()
            resumed = _configure_model(tmp, restored_dataset,
                                       loadcheck=stem + '.pt', start_from=1)
            resumed.args.sap_weight_commit = 'direction_only'
            resumed._current_task = 1
            resumed, _ = mammoth_load_checkpoint(stem + '.pt', resumed, args=resumed.args)
            self.assertEqual(resumed.sap_history[-1]['sap_weight_commit'], 'direction_only')
            for key, value in committed.items():
                torch.testing.assert_close(resumed.net.state_dict()[key], value, rtol=0, atol=0)
                torch.testing.assert_close(resumed.past_model_ckpt[key], value, rtol=0, atol=0)
            # Exercise the existing AER restore at the next task's fitting epoch.
            with torch.no_grad():
                resumed.net.classifier.weight.add_(1)
            resumed.begin_epoch(1, restored_dataset)
            for key, value in committed.items():
                torch.testing.assert_close(resumed.net.state_dict()[key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
