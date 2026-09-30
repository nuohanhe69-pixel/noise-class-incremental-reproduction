"""Normalized cosine CE applies only to observe's current and replay losses."""

import argparse
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from backbone.ResNetBlock import resnet18
from models.aer_sap import AerSap


class _FeatureNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = nn.Linear(4, 4, bias=False)
        self.classifier = nn.Linear(4, num_classes)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.tensor([
                [1., 0., 0., 0.], [0.5, 1., 0., 0.],
                [0., 0., 1., 0.], [0., 0., 0.5, 1.],
            ]))
            self.classifier.weight.copy_(torch.tensor([
                [2., 0., 0., 0.], [0., 3., 0., 0.],
                [0., 0., 4., 0.], [0., 0., 0., 5.],
                [1., 1., 0., 0.], [0., 0., 1., 1.],
            ])[:num_classes])
            self.classifier.bias.copy_(
                torch.tensor([8., -6., 4., -2., 3., -3.])[:num_classes],
            )

    def forward(self, inputs, returnt='out'):
        features = self.backbone(inputs)
        if returnt == 'features':
            return features
        return self.classifier(features)


class _Buffer:
    def __init__(self, replay=None):
        self.replay = replay
        self.updated_scores = None
        self.inserted = None
        self.sample_selection_fn = SimpleNamespace(update=self._update)

    def _update(self, indexes, losses):
        self.updated_scores = (indexes.clone(), losses.detach().clone())

    def is_empty(self):
        return self.replay is None

    def get_data(self, *_args, **_kwargs):
        return self.replay

    def add_data(self, **kwargs):
        self.inserted = kwargs


def _cosine_logits(net, inputs):
    features = net.backbone(inputs)
    return F.normalize(features, p=2, dim=1) @ F.normalize(
        net.classifier.weight, p=2, dim=1,
    ).T


def _model(training_loss, *, task=0, replay=None, num_classes=4):
    model = AerSap.__new__(AerSap)
    nn.Module.__init__(model)
    model.net = _FeatureNet(num_classes)
    model.args = SimpleNamespace(
        training_loss=training_loss, cosine_inference=0,
        use_aer=1, n_epochs=3, minibatch_size=1,
        sample_selection_strategy='abs', alpha_sample_insertion=0.0,
    )
    model.num_classes = num_classes
    model._current_task = task
    model._n_seen_classes = 2 if task == 0 else 4
    model.seen_so_far = torch.empty(0, dtype=torch.long) if task == 0 else torch.tensor([0, 1])
    model.opt = torch.optim.SGD(model.net.parameters(), lr=0.0)
    model.transform = nn.Identity()
    model.normalization_transform = nn.Identity()
    model.buffer = _Buffer(replay)
    model.loss_trace_recorder = None
    calls = []

    def loss(logits, labels, reduction='mean'):
        calls.append((logits.detach().clone(), labels.detach().clone(), reduction))
        return F.cross_entropy(logits, labels, reduction=reduction)

    model.loss = loss
    return model, calls


class AerSapNormalizedCosineCETests(unittest.TestCase):
    def test_resnet18_feature_branch_is_classifier_input(self):
        net = resnet18(100)
        net.eval()
        captured = []
        handle = net.classifier.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0]),
        )
        with torch.no_grad():
            inputs = torch.randn(1, 3, 32, 32)
            features = net(inputs, returnt='features')
            net(inputs)
        handle.remove()
        self.assertEqual(tuple(features.shape), (1, net.classifier.in_features))
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(features, captured[0]))
        model, _ = _model('normalized_cosine_ce')
        model.net = net
        with torch.no_grad():
            logits = model._observe_training_logits(inputs)
            expected = F.normalize(features, p=2, dim=1) @ F.normalize(
                net.classifier.weight, p=2, dim=1,
            ).T
        self.assertEqual(tuple(logits.shape), (1, 100))
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_cli_default_and_original_training_logits(self):
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        self.assertEqual(parser.parse_args(['--buffer_size', '4']).training_loss, 'ce')
        self.assertEqual(parser.parse_args([
            '--buffer_size', '4', '--training_loss', 'normalized_cosine_ce',
        ]).training_loss, 'normalized_cosine_ce')
        model, _ = _model('ce')
        inputs = torch.tensor([[1., 2., 3., 4.]])
        torch.testing.assert_close(model._observe_training_logits(inputs),
                                   model.net(inputs), rtol=0, atol=0)

    def test_full_cosine_logits_ignore_bias_without_normalizing_parameters(self):
        model, _ = _model('normalized_cosine_ce')
        inputs = torch.tensor([[1., 2., 3., 4.], [4., 1., 2., 3.]])
        weight_before = model.net.classifier.weight.detach().clone()
        logits = model._observe_training_logits(inputs)
        self.assertEqual(tuple(logits.shape), (2, 4))
        torch.testing.assert_close(logits, _cosine_logits(model.net, inputs), rtol=0, atol=0)
        with torch.no_grad():
            model.net.classifier.bias.fill_(float('nan'))
        torch.testing.assert_close(model._observe_training_logits(inputs), logits,
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(model.net.classifier.weight, weight_before))

    def test_default_observe_keeps_linear_current_and_replay_losses(self):
        current = torch.tensor([[1., 2., 3., 4.]])
        memory = torch.tensor([[4., 3., 2., 1.]])
        replay = (torch.tensor([0]), memory, memory, torch.tensor([0]))
        model, calls = _model('ce', task=1, replay=replay)
        reference_net = copy.deepcopy(model.net)
        model.observe(current, torch.tensor([2]), current, epoch=1,
                      true_labels=torch.tensor([2]), sample_ids=torch.tensor([20]))
        linear_current = reference_net(current)
        linear_current[:, :2] = torch.finfo(linear_current.dtype).min
        torch.testing.assert_close(calls[1][0], linear_current, rtol=0, atol=0)
        torch.testing.assert_close(calls[3][0], reference_net(memory), rtol=0, atol=0)

    def test_task0_current_loss_uses_cosine_but_insertion_score_uses_linear_ce(self):
        model, calls = _model('normalized_cosine_ce')
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        not_aug = torch.tensor([[1., 1., 3., 4.], [2., 2., 4., 3.]])
        labels = torch.tensor([0, 1])
        model.observe(inputs, labels, not_aug, epoch=0, true_labels=labels,
                      sample_ids=torch.tensor([10, 11]))
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0][0], model.net(not_aug), rtol=0, atol=0)
        current_cosine = _cosine_logits(model.net, inputs)
        current_cosine[:, 2:] = torch.finfo(current_cosine.dtype).min
        torch.testing.assert_close(calls[1][0], current_cosine, rtol=0, atol=0)
        self.assertEqual([call[2] for call in calls], ['none', 'mean'])
        scores = F.cross_entropy(model.net(not_aug), labels, reduction='none')
        torch.testing.assert_close(model.buffer.inserted['sample_selection_scores'],
                                   scores, rtol=0, atol=0)
        self.assertIsNone(model.net.classifier.bias.grad)

    def test_current_present_only_and_replay_seen_only_gradients(self):
        model, _ = _model('normalized_cosine_ce', task=1, num_classes=6)
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        current_labels = torch.tensor([2, 3])
        full_logits = _cosine_logits(model.net, inputs).detach()
        current_logits = model._observe_training_logits(
            inputs, present=current_labels.unique(),
        )
        self.assertEqual(tuple(current_logits.shape), (2, 6))
        torch.testing.assert_close(current_logits[:, 2:4], full_logits[:, 2:4],
                                   rtol=0, atol=0)
        excluded = torch.cat((current_logits[:, :2], current_logits[:, 4:]), dim=1)
        self.assertTrue(torch.equal(excluded, torch.full_like(
            excluded, torch.finfo(current_logits.dtype).min,
        )))
        F.cross_entropy(current_logits, current_labels).backward()
        current_grad = model.net.classifier.weight.grad.clone()
        self.assertTrue(torch.equal(current_grad[:2], torch.zeros_like(current_grad[:2])))
        self.assertGreater(current_grad[2:4].abs().sum().item(), 0)
        self.assertTrue(torch.equal(current_grad[4:], torch.zeros_like(current_grad[4:])))

        model.net.zero_grad(set_to_none=True)
        replay_labels = torch.tensor([0, 2])
        replay_logits = model._observe_training_logits(inputs, replay=True)
        self.assertEqual(tuple(replay_logits.shape), (2, 6))
        torch.testing.assert_close(replay_logits[:, :4], full_logits[:, :4],
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(replay_logits[:, 4:], torch.full_like(
            replay_logits[:, 4:], torch.finfo(replay_logits.dtype).min,
        )))
        F.cross_entropy(replay_logits, replay_labels).backward()
        replay_grad = model.net.classifier.weight.grad
        self.assertGreater(replay_grad[:4].abs().sum().item(), 0)
        self.assertTrue(torch.equal(replay_grad[4:], torch.zeros_like(replay_grad[4:])))

    def test_task1_current_and_replay_backprop_cosine_with_original_scoring_and_mask(self):
        current = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        not_aug_current = torch.tensor([[1., 1., 3., 4.], [2., 2., 4., 3.]])
        memory = torch.tensor([[4., 3., 2., 1.]])
        not_aug_memory = torch.tensor([[3., 3., 1., 2.]])
        memory_indexes = torch.tensor([7])
        replay = (memory_indexes, not_aug_memory, memory, torch.tensor([0]))
        model, calls = _model('normalized_cosine_ce', task=1, replay=replay,
                              num_classes=6)
        reference_net = copy.deepcopy(model.net)
        weight_before = model.net.classifier.weight.detach().clone()
        current_labels = torch.tensor([2, 3])
        loss = model.observe(current, current_labels, not_aug_current, epoch=1,
                             true_labels=current_labels, sample_ids=torch.tensor([20, 21]))
        self.assertEqual(len(calls), 4)
        current_cosine = _cosine_logits(reference_net, current)
        scoring_mask = torch.zeros_like(current_cosine)
        scoring_mask[:, 2:] = 1
        masked_cosine = current_cosine.clone()
        masked_cosine[:, :2] = torch.finfo(current_cosine.dtype).min
        masked_cosine[:, 4:] = torch.finfo(current_cosine.dtype).min
        current_linear = reference_net(not_aug_current).masked_fill(
            scoring_mask == 0, torch.finfo(current_cosine.dtype).min,
        )
        memory_linear = reference_net(not_aug_memory)
        memory_cosine = _cosine_logits(reference_net, memory)
        memory_cosine[:, 4:] = torch.finfo(memory_cosine.dtype).min
        for actual, expected in zip(calls, (
            current_linear, masked_cosine, memory_linear, memory_cosine,
        )):
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        self.assertEqual([call[2] for call in calls], ['none', 'mean', 'none', 'mean'])
        self.assertTrue(torch.equal(calls[1][0][:, :2],
                                    torch.full_like(calls[1][0][:, :2],
                                                    torch.finfo(current_cosine.dtype).min)))
        self.assertTrue(torch.equal(calls[1][0][:, 4:],
                                    torch.full_like(calls[1][0][:, 4:],
                                                    torch.finfo(current_cosine.dtype).min)))
        torch.testing.assert_close(calls[3][0], memory_cosine, rtol=0, atol=0)
        expected_score = F.cross_entropy(memory_linear, torch.tensor([0]),
                                         reduction='none')
        self.assertTrue(torch.equal(model.buffer.updated_scores[0], memory_indexes))
        torch.testing.assert_close(model.buffer.updated_scores[1], expected_score,
                                   rtol=0, atol=0)
        expected_loss = F.cross_entropy(masked_cosine, current_labels) + F.cross_entropy(
            memory_cosine, torch.tensor([0]),
        )
        expected_loss.backward()
        self.assertAlmostEqual(loss, expected_loss.item(), places=6)
        torch.testing.assert_close(model.net.backbone.weight.grad,
                                   reference_net.backbone.weight.grad)
        torch.testing.assert_close(model.net.classifier.weight.grad,
                                   reference_net.classifier.weight.grad)
        self.assertTrue(torch.equal(model.net.classifier.weight.grad[4:],
                                    torch.zeros_like(model.net.classifier.weight.grad[4:])))
        self.assertIsNone(model.net.classifier.bias.grad)
        self.assertTrue(torch.equal(model.net.classifier.weight, weight_before))
        self.assertIsNone(model.buffer.inserted)


if __name__ == '__main__':
    unittest.main()
