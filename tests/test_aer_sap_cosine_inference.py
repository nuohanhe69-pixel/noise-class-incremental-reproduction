"""Evaluation-only cosine logits from the Linear classifier input."""

import argparse
import copy
import json
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.aer_sap import AerSap
from tests.test_aer_sap import _Dataset, _model
from utils.evaluate import evaluate


class _EvaluationDataset(_Dataset):
    def __init__(self):
        super().__init__()
        self.test_loaders = self.test_loaders[:2]
        self.c_task = 1

    def get_offsets(self, task_id=None):
        return super().get_offsets(self.c_task if task_id is None else task_id)

    @staticmethod
    def get_loss():
        return nn.CrossEntropyLoss()

    def evaluate(self, model, _dataset):
        return evaluate(model, self)


def _offline_cosine(features, weight, n_seen_classes):
    seen_weight = weight[:n_seen_classes]
    scale = seen_weight.norm(p=2, dim=1).mean()
    return scale * torch.matmul(
        F.normalize(features, p=2, dim=1),
        F.normalize(seen_weight, p=2, dim=1).T,
    )


class AerSapCosineInferenceTests(unittest.TestCase):
    def test_cli_defaults_off_and_training_forward_stays_linear(self):
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        self.assertEqual(parser.parse_args(['--buffer_size', '20']).cosine_inference, 0)
        self.assertEqual(parser.parse_args([
            '--buffer_size', '20', '--cosine_inference', '1',
        ]).cosine_inference, 1)

        with tempfile.TemporaryDirectory() as directory:
            model = _model(directory)
            inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            model.net.eval()
            self.assertTrue(torch.equal(model(inputs), model.net(inputs)))
            model.args.cosine_inference = 1
            model.net.train()
            self.assertTrue(torch.equal(model(inputs), model.net(inputs)))

    def test_seen_rows_classifier_input_and_bias_free_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _model(directory)
            model._n_seen_classes = 20
            model.args.cosine_inference = 1
            with torch.no_grad():
                model.net.backbone.weight[0, 1] = 0.5
                model.net.classifier.bias.fill_(50.0)
            model.net.eval()
            inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 3.0]])
            state_before = copy.deepcopy(model.net.state_dict())
            logits = model(inputs)
            features = model.net.backbone(inputs)
            expected = _offline_cosine(features, model.net.classifier.weight, 20)
            self.assertEqual(tuple(logits.shape), (2, 20))
            torch.testing.assert_close(logits, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(model.net(inputs)[:, :20], logits))
            with torch.no_grad():
                model.net.classifier.bias.add_(1000)
                model.net.classifier.weight[20:].add_(1000)
            torch.testing.assert_close(model(inputs), logits, rtol=0, atol=0)
            self.assertTrue(torch.equal(
                model.net.backbone.weight, state_before['backbone.weight'],
            ))
            self.assertTrue(torch.equal(
                model.net.classifier.weight[:20], state_before['classifier.weight'][:20],
            ))

    def test_task1_artifact_matches_offline_cosine_class_and_task_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _model(directory)
            model._current_task = 1
            model._n_seen_classes = 20
            model.args.cosine_inference = 1
            model.args.eval_future = False
            dataset = _EvaluationDataset()
            model._run_taskwise_sap(dataset)
            artifact = model._taskwise_artifact_directory(dataset)
            manifest = json.loads((artifact / 'manifest.json').read_text())
            self.assertTrue(manifest['valid'])
            self.assertEqual(manifest['boundary_task_id'], 1)
            for stage, weight_name in (('pre_sap', 'W_before.pt'),
                                       ('post_sap', 'W_after.pt')):
                weight = torch.load(artifact / weight_name, weights_only=True)
                features = torch.load(artifact / f'test_{stage}_raw_features.pt',
                                      weights_only=True)
                saved_logits = torch.load(artifact / f'test_{stage}_logits.pt',
                                          weights_only=True)
                offline_logits = _offline_cosine(features, weight, 20)
                torch.testing.assert_close(
                    saved_logits, offline_logits, rtol=1e-6, atol=1e-6,
                )
            labels = torch.load(artifact / 'test_post_sap_labels.pt', weights_only=True)
            task_ids = torch.load(artifact / 'test_post_sap_task_ids.pt',
                                  weights_only=True)
            class_predictions = offline_logits.argmax(dim=1)
            self.assertTrue(torch.equal(
                class_predictions,
                torch.load(artifact / 'test_post_sap_predictions.pt', weights_only=True),
            ))
            task_predictions = torch.empty_like(class_predictions)
            for task_id in (0, 1):
                start, end = dataset.get_offsets(task_id)
                mask = task_ids == task_id
                task_predictions[mask] = offline_logits[mask, start:end].argmax(1) + start
            self.assertTrue(torch.equal(
                task_predictions,
                torch.load(artifact / 'test_post_sap_task_predictions.pt', weights_only=True),
            ))
            accuracy = json.loads((artifact / 'accuracy.json').read_text())['post_sap']
            for task_id in (0, 1):
                mask = task_ids == task_id
                class_accuracy = float((class_predictions[mask] == labels[mask]).float().mean() * 100)
                task_accuracy = float((task_predictions[mask] == labels[mask]).float().mean() * 100)
                self.assertEqual(accuracy['per_task_class_il'][task_id], class_accuracy)
                self.assertEqual(accuracy['per_task_task_il'][task_id], task_accuracy)
            self.assertEqual(accuracy['class_il'],
                             sum(accuracy['per_task_class_il']) / 2)
            self.assertEqual(accuracy['task_il'],
                             sum(accuracy['per_task_task_il']) / 2)


if __name__ == '__main__':
    unittest.main()
