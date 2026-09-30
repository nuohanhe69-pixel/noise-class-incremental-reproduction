"""Normalized cosine CE and sample scores share the active-class geometry."""

import argparse
import copy
import contextlib
import io
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


class _Shift(nn.Module):
    def forward(self, inputs):
        return inputs + torch.tensor([2., 0., 0., 0.])


def _cosine_logits(net, inputs):
    features = net.backbone(inputs)
    return F.normalize(features, p=2, dim=1) @ F.normalize(
        net.classifier.weight, p=2, dim=1,
    ).T


def _scoped_cosine_logits(net, inputs, scale, active_classes):
    logits = _cosine_logits(net, inputs)
    if scale != 1:
        logits = logits * scale
    inactive = torch.ones(logits.shape[1], dtype=torch.bool)
    inactive[active_classes] = False
    return logits.masked_fill(inactive, torch.finfo(logits.dtype).min)


def _model(training_loss, *, task=0, replay=None, num_classes=4, scoring_scale=1):
    model = AerSap.__new__(AerSap)
    nn.Module.__init__(model)
    model.net = _FeatureNet(num_classes)
    model.args = SimpleNamespace(
        training_loss=training_loss, cosine_inference=0,
        scale_cosine_scoring_scale=scoring_scale,
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
        self.assertEqual(parser.parse_args(['--buffer_size', '4']).scale_cosine_scoring_scale, 1)
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
        torch.testing.assert_close(calls[0][0], linear_current, rtol=0, atol=0)
        torch.testing.assert_close(calls[1][0], linear_current, rtol=0, atol=0)
        torch.testing.assert_close(calls[2][0], reference_net(memory), rtol=0, atol=0)
        torch.testing.assert_close(calls[3][0], reference_net(memory), rtol=0, atol=0)
        expected_score = F.cross_entropy(reference_net(memory), torch.tensor([0]),
                                         reduction='none')
        torch.testing.assert_close(model.buffer.updated_scores[1], expected_score,
                                   rtol=0, atol=0)

    def test_task0_current_scoring_and_insertion_use_present_only_cosine(self):
        model, calls = _model('normalized_cosine_ce')
        model.args.alpha_sample_insertion = 0.75
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.],
                               [3., 2., 1., 4.], [1., 4., 2., 3.]])
        not_aug = torch.tensor([[1., 1., 3., 4.], [2., 2., 4., 3.],
                                [4., 1., 2., 3.], [2., 3., 1., 4.]])
        labels = torch.tensor([0, 1, 0, 1])
        sample_ids = torch.tensor([10, 11, 12, 13])
        model.observe(inputs, labels, not_aug, epoch=0, true_labels=labels,
                      sample_ids=sample_ids)
        self.assertEqual(len(calls), 2)
        scoring_cosine = _cosine_logits(model.net, not_aug)
        scoring_cosine[:, 2:] = torch.finfo(scoring_cosine.dtype).min
        torch.testing.assert_close(calls[0][0], scoring_cosine, rtol=0, atol=0)
        current_cosine = _cosine_logits(model.net, inputs)
        current_cosine[:, 2:] = torch.finfo(current_cosine.dtype).min
        torch.testing.assert_close(calls[1][0], current_cosine, rtol=0, atol=0)
        self.assertEqual([call[2] for call in calls], ['none', 'mean'])
        scores = F.cross_entropy(scoring_cosine, labels, reduction='none')
        selected = torch.topk(scores, 1, largest=False).indices
        torch.testing.assert_close(model.buffer.inserted['sample_selection_scores'],
                                   scores[selected], rtol=0, atol=0)
        torch.testing.assert_close(model.buffer.inserted['examples'], not_aug[selected],
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(model.buffer.inserted['labels'], labels[selected]))
        self.assertTrue(torch.equal(model.buffer.inserted['sample_ids'], sample_ids[selected]))
        self.assertIsNone(model.net.classifier.bias.grad)

    def test_scoring_ignores_bias_and_normalizes_not_aug_inputs(self):
        model, calls = _model('normalized_cosine_ce')
        model.normalization_transform = _Shift()
        with torch.no_grad():
            model.net.classifier.bias.fill_(float('nan'))
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        not_aug = torch.tensor([[1., 1., 3., 4.], [2., 2., 4., 3.]])
        labels = torch.tensor([0, 1])
        model.observe(inputs, labels, not_aug, epoch=0)
        expected = _cosine_logits(model.net, not_aug + torch.tensor([2., 0., 0., 0.]))
        expected[:, 2:] = torch.finfo(expected.dtype).min
        torch.testing.assert_close(calls[0][0], expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(calls[0][0]).all())
        self.assertTrue(torch.isfinite(model.buffer.inserted['sample_selection_scores']).all())

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

    def test_task1_current_and_replay_scores_match_training_geometry(self):
        current = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        not_aug_current = torch.tensor([[1., 1., 3., 4.], [2., 2., 4., 3.]])
        memory = torch.tensor([[4., 3., 2., 1.]])
        not_aug_memory = torch.tensor([[3., 3., 1., 2.]])
        memory_indexes = torch.tensor([7])
        replay = (memory_indexes, not_aug_memory, memory, torch.tensor([0]))
        model, calls = _model('normalized_cosine_ce', task=1, replay=replay,
                              num_classes=6)
        reference_net = copy.deepcopy(model.net)
        with torch.no_grad():
            model.net.classifier.bias.fill_(float('nan'))
        weight_before = model.net.classifier.weight.detach().clone()
        current_labels = torch.tensor([2, 3])
        loss = model.observe(current, current_labels, not_aug_current, epoch=1,
                             true_labels=current_labels, sample_ids=torch.tensor([20, 21]))
        self.assertEqual(len(calls), 4)
        current_cosine = _cosine_logits(reference_net, current)
        masked_cosine = current_cosine.clone()
        masked_cosine[:, :2] = torch.finfo(current_cosine.dtype).min
        masked_cosine[:, 4:] = torch.finfo(current_cosine.dtype).min
        current_scoring = _cosine_logits(reference_net, not_aug_current)
        current_scoring[:, :2] = torch.finfo(current_scoring.dtype).min
        current_scoring[:, 4:] = torch.finfo(current_scoring.dtype).min
        memory_scoring = _cosine_logits(reference_net, not_aug_memory)
        memory_scoring[:, 4:] = torch.finfo(memory_scoring.dtype).min
        memory_cosine = _cosine_logits(reference_net, memory)
        memory_cosine[:, 4:] = torch.finfo(memory_cosine.dtype).min
        for actual, expected in zip(calls, (
            current_scoring, masked_cosine, memory_scoring, memory_cosine,
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
        expected_score = F.cross_entropy(memory_scoring, torch.tensor([0]),
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

    def test_scale_cosine_parser_accepts_only_scoring_scale_1_or_64(self):
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        for scoring_scale in (1, 64):
            parsed = parser.parse_args([
                '--buffer_size', '4', '--training_loss', 'scale_cosine_ce',
                '--scale_cosine_scoring_scale', str(scoring_scale),
            ])
            self.assertEqual(parsed.scale_cosine_scoring_scale, scoring_scale)
        with contextlib.redirect_stderr(io.StringIO()):
            for invalid in ('0', '16', '32', '128'):
                with self.assertRaises(SystemExit):
                    parser.parse_args([
                        '--buffer_size', '4', '--scale_cosine_scoring_scale', invalid,
                    ])

    def test_scale_cosine_training_scope_and_future_row_gradients(self):
        model, _ = _model('scale_cosine_ce', task=1, num_classes=6)
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        labels = torch.tensor([2, 3])
        original_weight = model.net.classifier.weight.detach().clone()
        with torch.no_grad():
            model.net.classifier.bias.fill_(float('nan'))

        current = model._observe_training_logits(inputs, present=labels.unique())
        expected_current = _scoped_cosine_logits(model.net, inputs, 64, labels.unique())
        torch.testing.assert_close(current, expected_current, rtol=0, atol=0)
        F.cross_entropy(current, labels).backward()
        current_grad = model.net.classifier.weight.grad.clone()
        self.assertTrue(torch.equal(current_grad[:2], torch.zeros_like(current_grad[:2])))
        self.assertGreater(current_grad[2:4].abs().sum().item(), 0)
        self.assertTrue(torch.equal(current_grad[4:], torch.zeros_like(current_grad[4:])))

        model.net.zero_grad(set_to_none=True)
        replay = model._observe_training_logits(inputs, replay=True)
        expected_replay = _scoped_cosine_logits(model.net, inputs, 64, slice(0, 4))
        torch.testing.assert_close(replay, expected_replay, rtol=0, atol=0)
        F.cross_entropy(replay, torch.tensor([0, 2])).backward()
        replay_grad = model.net.classifier.weight.grad
        self.assertGreater(replay_grad[:4].abs().sum().item(), 0)
        self.assertTrue(torch.equal(replay_grad[4:], torch.zeros_like(replay_grad[4:])))
        self.assertIsNone(model.net.classifier.bias.grad)
        self.assertTrue(torch.equal(model.net.classifier.weight, original_weight))

    def test_scale_cosine_e1_e2_share_training_loss_and_gradients(self):
        current = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        not_aug_current = torch.tensor([[1., 1., 3., 4.], [2., 2., 4., 3.]])
        memory = torch.tensor([[4., 3., 2., 1.]])
        not_aug_memory = torch.tensor([[3., 3., 1., 2.]])
        replay = (torch.tensor([7]), not_aug_memory, memory, torch.tensor([0]))
        labels = torch.tensor([2, 3])
        outputs = []
        for scoring_scale in (1, 64):
            model, calls = _model('scale_cosine_ce', task=1, replay=replay,
                                  num_classes=6, scoring_scale=scoring_scale)
            weight_before = model.net.classifier.weight.detach().clone()
            with torch.no_grad():
                model.net.classifier.bias.fill_(float('nan'))
            loss = model.observe(current, labels, not_aug_current, epoch=1)
            expected_current_scoring = _scoped_cosine_logits(
                model.net, not_aug_current, scoring_scale, labels.unique(),
            )
            expected_replay_scoring = _scoped_cosine_logits(
                model.net, not_aug_memory, scoring_scale, slice(0, 4),
            )
            expected_current_training = _scoped_cosine_logits(
                model.net, current, 64, labels.unique(),
            )
            expected_replay_training = _scoped_cosine_logits(
                model.net, memory, 64, slice(0, 4),
            )
            for actual, expected in zip(calls, (
                expected_current_scoring, expected_current_training,
                expected_replay_scoring, expected_replay_training,
            )):
                torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
            self.assertEqual([call[2] for call in calls], ['none', 'mean', 'none', 'mean'])
            expected_score = F.cross_entropy(expected_replay_scoring, replay[3],
                                             reduction='none')
            self.assertTrue(torch.equal(model.buffer.updated_scores[0], replay[0]))
            torch.testing.assert_close(model.buffer.updated_scores[1], expected_score,
                                       rtol=0, atol=0)
            self.assertIsNone(model.buffer.inserted)
            self.assertIsNone(model.net.classifier.bias.grad)
            self.assertTrue(torch.equal(model.net.classifier.weight, weight_before))
            outputs.append((model, calls, loss))

        e1_model, e1_calls, e1_loss = outputs[0]
        e2_model, e2_calls, e2_loss = outputs[1]
        self.assertEqual(e1_loss, e2_loss)
        for index in (1, 3):
            self.assertTrue(torch.equal(e1_calls[index][0], e2_calls[index][0]))
        for parameter_name in ('backbone.weight', 'classifier.weight'):
            e1_parameter = dict(e1_model.net.named_parameters())[parameter_name]
            e2_parameter = dict(e2_model.net.named_parameters())[parameter_name]
            self.assertTrue(torch.equal(e1_parameter.grad, e2_parameter.grad))
            self.assertTrue(torch.equal(e1_parameter, e2_parameter))
        self.assertFalse(torch.equal(e1_model.buffer.updated_scores[1],
                                     e2_model.buffer.updated_scores[1]))

    def test_scale_cosine_insertion_scores_follow_selected_scoring_scale(self):
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.],
                               [3., 2., 1., 4.], [1., 4., 2., 3.]])
        not_aug = torch.tensor([[1., 1., 3., 4.], [2., 2., 4., 3.],
                                [4., 1., 2., 3.], [2., 3., 1., 4.]])
        labels = torch.tensor([0, 1, 0, 1])
        sample_ids = torch.tensor([10, 11, 12, 13])
        for scoring_scale in (1, 64):
            with self.subTest(scoring_scale=scoring_scale):
                model, calls = _model('scale_cosine_ce', scoring_scale=scoring_scale)
                model.args.alpha_sample_insertion = 0.75
                model.observe(inputs, labels, not_aug, epoch=0,
                              true_labels=labels, sample_ids=sample_ids)
                scoring_logits = _scoped_cosine_logits(
                    model.net, not_aug, scoring_scale, labels.unique(),
                )
                training_logits = _scoped_cosine_logits(
                    model.net, inputs, 64, labels.unique(),
                )
                torch.testing.assert_close(calls[0][0], scoring_logits, rtol=0, atol=0)
                torch.testing.assert_close(calls[1][0], training_logits, rtol=0, atol=0)
                scores = F.cross_entropy(scoring_logits, labels, reduction='none')
                selected = torch.topk(scores, 1, largest=False).indices
                inserted = model.buffer.inserted
                self.assertEqual(len(inserted['examples']), 1)
                torch.testing.assert_close(inserted['sample_selection_scores'],
                                           scores[selected], rtol=0, atol=0)
                self.assertTrue(torch.equal(inserted['sample_ids'], sample_ids[selected]))
                self.assertTrue(torch.equal(inserted['labels'], labels[selected]))
                torch.testing.assert_close(inserted['examples'], not_aug[selected],
                                           rtol=0, atol=0)

    def test_existing_modes_ignore_scale_cosine_scoring_option(self):
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        for training_loss in ('ce', 'normalized_cosine_ce'):
            models = [_model(training_loss, task=1, num_classes=6,
                             scoring_scale=scale)[0] for scale in (1, 64)]
            for replay, present in ((False, torch.tensor([2, 3])), (True, None)):
                training = [model._observe_training_logits(
                    inputs, present=present, replay=replay,
                ) for model in models]
                scoring = [model._observe_scoring_logits(
                    inputs, present=present, replay=replay,
                ) for model in models]
                self.assertTrue(torch.equal(training[0], training[1]))
                self.assertTrue(torch.equal(scoring[0], scoring[1]))
                if training_loss == 'ce':
                    self.assertTrue(torch.equal(training[0], models[0].net(inputs)))
                    self.assertTrue(torch.equal(scoring[0], models[0].net(inputs)))
                else:
                    expected = _scoped_cosine_logits(
                        models[0].net, inputs, 1,
                        present if present is not None else slice(0, 4),
                    )
                    torch.testing.assert_close(training[0], expected, rtol=0, atol=0)
                    torch.testing.assert_close(scoring[0], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
