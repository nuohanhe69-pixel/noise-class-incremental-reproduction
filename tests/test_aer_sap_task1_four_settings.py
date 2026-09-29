"""Task1 four-setting report stays separate from the current-task SAP state."""

import copy
import csv
import json
import random
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from tests.test_aer_sap import _Dataset, _model


SETTINGS = ('none', 'task0_only', 'task1_only', 'both')
COLUMNS = (
    'setting', 'Task0 Class-IL', 'Task1 Class-IL', 'Overall Class-IL',
    'Task0 Task-IL', 'Task1 Task-IL', 'Overall Task-IL',
)


def _load(directory, filename):
    return torch.load(directory / filename, weights_only=True)


def _offline_cosine(features, weight):
    seen = weight[:20]
    scale = seen.norm(p=2, dim=1).mean()
    return scale * (F.normalize(features, p=2, dim=1) @ F.normalize(
        seen, p=2, dim=1,
    ).T)


class AerSapTask1FourSettingsTests(unittest.TestCase):
    def test_task1_report_uses_new_old_projector_and_preserves_formal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _model(directory)
            dataset = _Dataset()
            model._run_taskwise_sap(dataset)
            task0_artifact = model._taskwise_artifact_directory(dataset)
            old_task0_matrix = _load(task0_artifact, 'M_task_0.pt')
            self.assertFalse((task0_artifact / 'task1_four_settings.json').exists())

            with torch.no_grad():
                model.net.backbone.weight[0, 1] += 0.5
            model._current_task = 1
            model._n_seen_classes = 20
            pre_state = copy.deepcopy(model.net.state_dict())
            torch_rng = torch.random.get_rng_state().clone()
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            model._run_taskwise_sap(dataset)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))
            self.assertEqual(random.getstate(), python_rng)
            self.assertTrue(all(np.array_equal(left, right) for left, right in
                                zip(np.random.get_state(), numpy_rng)))
            self.assertFalse(hasattr(model.args, 'cosine_inference'))

            artifact = model._taskwise_artifact_directory(dataset)
            manifest = json.loads((artifact / 'manifest.json').read_text())
            self.assertEqual(set(manifest['files']),
                             {path.name for path in artifact.iterdir()} - {'manifest.json'})
            self.assertNotIn('M_task_0.pt', manifest['files'])
            self.assertIn('counterfactual_M_task_0.pt', manifest['files'])
            self.assertIn('M_task_1.pt', manifest['files'])
            self.assertEqual(manifest['projection_scope'], 'current_task_only')

            before = _load(artifact, 'W_before.pt')
            after = _load(artifact, 'W_after.pt')
            old_matrix = _load(artifact, 'counterfactual_M_task_0.pt')
            new_matrix = _load(artifact, 'M_task_1.pt')
            reference_features = _load(artifact, 'X_l2.pt')
            reference_tasks = _load(artifact, 'reference_source_task_ids.pt')
            historical_features = reference_features[reference_tasks == 0]
            self.assertGreater(len(historical_features), 0)
            old_gram = historical_features.T @ historical_features
            torch.testing.assert_close(_load(artifact, 'counterfactual_G_task_0.pt'),
                                       old_gram, rtol=0, atol=0)
            expected_matrix, _, _, _, _, expected_eigenvalues = (
                model._build_oracle_projection(old_gram, return_eigenvectors=True)
            )
            torch.testing.assert_close(old_matrix, expected_matrix)
            torch.testing.assert_close(
                _load(artifact, 'counterfactual_eigenvalues_task_0.pt'),
                expected_eigenvalues,
            )
            eigenvectors = _load(artifact, 'counterfactual_eigenvectors_task_0.pt')
            eigenvalues = _load(artifact, 'counterfactual_eigenvalues_task_0.pt')
            torch.testing.assert_close(old_gram,
                                       (eigenvectors * eigenvalues) @ eigenvectors.T)
            self.assertFalse(torch.equal(old_matrix, old_task0_matrix))
            self.assertTrue(torch.equal(before, pre_state['classifier.weight']))

            weights = {
                name: _load(artifact, f'task1_four_settings_{name}_W.pt')
                for name in SETTINGS
            }
            self.assertTrue(torch.equal(weights['none'], before))
            self.assertTrue(torch.equal(weights['task1_only'], after))
            torch.testing.assert_close(weights['task0_only'][:10],
                                       before[:10] @ old_matrix.T)
            self.assertTrue(torch.equal(weights['task0_only'][10:], before[10:]))
            torch.testing.assert_close(weights['task1_only'][10:20],
                                       before[10:20] @ new_matrix.T)
            self.assertTrue(torch.equal(weights['task1_only'][:10], before[:10]))
            self.assertTrue(torch.equal(weights['task1_only'][20:], before[20:]))
            self.assertTrue(torch.equal(weights['both'][:10], weights['task0_only'][:10]))
            self.assertTrue(torch.equal(weights['both'][10:20], after[10:20]))
            self.assertTrue(torch.equal(weights['both'][20:], before[20:]))

            report = json.loads((artifact / 'task1_four_settings.json').read_text())
            self.assertEqual(report['projection_scope'], 'analysis_only')
            self.assertEqual(report['inference'], 'cosine')
            self.assertEqual([row['setting'] for row in report['settings']], list(SETTINGS))
            self.assertEqual(set(report['settings'][0]), set(COLUMNS))
            with (artifact / 'task1_four_settings.csv').open(newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual([row['setting'] for row in csv_rows], list(SETTINGS))
            self.assertEqual(tuple(csv_rows[0]), COLUMNS)
            self.assertIn('| ' + ' | '.join(COLUMNS) + ' |',
                          (artifact / 'task1_four_settings.md').read_text())

            features = _load(artifact, 'test_pre_sap_raw_features.pt')
            identities = None
            for name, row in zip(SETTINGS, report['settings']):
                prefix = f'task1_four_settings_{name}'
                actual_ids = tuple(_load(artifact, f'{prefix}_{field}.pt') for field in (
                    'sample_ids', 'true_labels', 'source_task_ids',
                ))
                if identities is None:
                    identities = actual_ids
                else:
                    for actual, expected in zip(actual_ids, identities):
                        self.assertTrue(torch.equal(actual, expected))
                logits = _load(artifact, f'{prefix}_logits.pt')
                torch.testing.assert_close(logits, _offline_cosine(features, weights[name]),
                                           rtol=1e-6, atol=1e-6)
                predictions = logits.argmax(1)
                self.assertTrue(torch.equal(
                    predictions, _load(artifact, f'{prefix}_predictions.pt'),
                ))
                task_predictions = torch.empty_like(predictions)
                for task_id in (0, 1):
                    mask = actual_ids[2] == task_id
                    start, end = dataset.get_offsets(task_id)
                    task_predictions[mask] = logits[mask, start:end].argmax(1) + start
                    class_accuracy = float((predictions[mask] == actual_ids[1][mask]).float().mean() * 100)
                    task_accuracy = float((task_predictions[mask] == actual_ids[1][mask]).float().mean() * 100)
                    self.assertEqual(row[f'Task{task_id} Class-IL'], class_accuracy)
                    self.assertEqual(row[f'Task{task_id} Task-IL'], task_accuracy)
                self.assertTrue(torch.equal(
                    task_predictions, _load(artifact, f'{prefix}_task_predictions.pt'),
                ))
                self.assertEqual(row['Overall Class-IL'],
                                 (row['Task0 Class-IL'] + row['Task1 Class-IL']) / 2)
                self.assertEqual(row['Overall Task-IL'],
                                 (row['Task0 Task-IL'] + row['Task1 Task-IL']) / 2)
            for field, formal in (
                ('sample_ids', 'sample_ids'), ('true_labels', 'labels'),
                ('source_task_ids', 'task_ids'),
            ):
                self.assertTrue(torch.equal(
                    identities[('sample_ids', 'true_labels', 'source_task_ids').index(field)],
                    _load(artifact, f'test_pre_sap_{formal}.pt'),
                ))

            self.assertTrue(torch.equal(_load(artifact, 'bias_before.pt'),
                                        _load(artifact, 'bias_after.pt')))
            self.assertTrue(torch.equal(model.net.classifier.weight, after))
            post_state = _load(artifact, 'network_post_sap.pt')
            for key, value in post_state.items():
                self.assertTrue(torch.equal(model.net.state_dict()[key], value))
                self.assertTrue(torch.equal(model.past_model_ckpt[key], value))
            self.assertEqual(set(json.loads((artifact / 'accuracy.json').read_text())),
                             {'pre_sap', 'post_sap'})


if __name__ == '__main__':
    unittest.main()
