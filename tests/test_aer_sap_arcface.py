"""ArcFace changes only observed-label targets in cosine training CE."""

import argparse
import math
import unittest

import torch
from torch.nn import functional as F

from models.aer_sap import AerSap, arcface_training_logits
from tests.test_aer_sap_normalized_cosine_ce import _model, _scoped_cosine_logits


def _arc_model(margin=0.5, **kwargs):
    model, calls = _model('arcface_ce', **kwargs)
    model.args.arcface_margin = margin
    return model, calls


class AerSapArcFaceTests(unittest.TestCase):
    def test_parser_keeps_defaults_and_all_existing_modes(self):
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        defaults = parser.parse_args(['--buffer_size', '4'])
        self.assertEqual(defaults.training_loss, 'ce')
        self.assertEqual(defaults.arcface_margin, 0.5)
        self.assertEqual(defaults.cosine_inference, 0)
        for mode in ('ce', 'normalized_cosine_ce', 'scale_cosine_ce', 'nce', 'arcface_ce'):
            parsed = parser.parse_args(['--buffer_size', '4', '--training_loss', mode])
            self.assertEqual(parsed.training_loss, mode)
        self.assertEqual(parser.parse_args([
            '--buffer_size', '4', '--training_loss', 'arcface_ce', '--arcface_margin', '0',
        ]).arcface_margin, 0)

    def test_margin_zero_matches_e1_current_and_replay_loss_and_gradients_exactly(self):
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        for replay, labels in ((False, torch.tensor([2, 3])), (True, torch.tensor([0, 2]))):
            with self.subTest(replay=replay):
                arc, _ = _arc_model(margin=0, task=1, num_classes=6)
                e1, _ = _model('scale_cosine_ce', task=1, num_classes=6, scoring_scale=1)
                kwargs = {'replay': True} if replay else {'present': labels.unique()}
                raw = arc._observe_training_logits(inputs, **kwargs)
                scaled = e1._observe_training_logits(inputs, **kwargs)
                self.assertTrue(torch.equal(arcface_training_logits(raw, labels, 0), scaled))
                arc_loss = arc._compute_training_loss(raw, labels)
                e1_loss = e1._compute_training_loss(scaled, labels)
                self.assertTrue(torch.equal(arc_loss, e1_loss))
                arc_loss.backward()
                e1_loss.backward()
                for name in ('backbone.weight', 'classifier.weight'):
                    arc_parameter = dict(arc.net.named_parameters())[name]
                    e1_parameter = dict(e1.net.named_parameters())[name]
                    self.assertTrue(torch.equal(arc_parameter.grad, e1_parameter.grad), name)
                self.assertIsNone(arc.net.classifier.bias.grad)

    def test_target_only_transform_monotonic_branch_and_mask(self):
        masked = torch.finfo(torch.float32).min
        cosine = torch.tensor([[0.3, 0.6, masked, masked],
                               [0.9, -0.95, 0.1, masked]], requires_grad=True)
        original = cosine.detach().clone()
        labels = torch.tensor([0, 1])
        logits = arcface_training_logits(cosine, labels)
        expected = original.clone()
        expected[0, 0] = math.cos(math.acos(0.3) + 0.5)
        expected[1, 1] = -0.95 - math.sin(math.pi - 0.5) * 0.5
        active = original != masked
        expected[active] *= 64
        torch.testing.assert_close(logits, expected)
        self.assertTrue(torch.equal(logits[~active], original[~active]))
        self.assertTrue(torch.equal(cosine.detach(), original))
        non_targets = active.clone()
        non_targets[torch.arange(2), labels] = False
        self.assertTrue(torch.equal(logits[non_targets], original[non_targets] * 64))
        with self.assertRaisesRegex(ValueError, 'target class must be active'):
            arcface_training_logits(cosine, torch.tensor([2, 1]))

    def test_cosine_boundaries_have_finite_loss_and_gradients(self):
        epsilon = torch.finfo(torch.float32).eps
        cosine = torch.tensor([[value, 0.2] for value in
                               (-1 - epsilon, -1, 1, 1 + epsilon)], requires_grad=True)
        logits = arcface_training_logits(cosine, torch.zeros(4, dtype=torch.long))
        loss = F.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(cosine.grad).all())

    def test_current_present_only_replay_seen_only_bias_free_and_future_gradients(self):
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        for replay, labels in ((False, torch.tensor([2, 3])), (True, torch.tensor([0, 2]))):
            with self.subTest(replay=replay):
                model, _ = _arc_model(task=1, num_classes=6)
                weight_before = model.net.classifier.weight.detach().clone()
                with torch.no_grad():
                    model.net.classifier.bias.fill_(float('nan'))
                kwargs = {'replay': True} if replay else {'present': labels.unique()}
                raw = model._observe_training_logits(inputs, **kwargs)
                expected = _scoped_cosine_logits(
                    model.net, inputs, 1, slice(0, 4) if replay else labels.unique(),
                )
                torch.testing.assert_close(raw, expected, rtol=0, atol=0)
                model._compute_training_loss(raw, labels).backward()
                gradient = model.net.classifier.weight.grad
                self.assertTrue(torch.equal(gradient[4:], torch.zeros_like(gradient[4:])))
                if not replay:
                    self.assertTrue(torch.equal(gradient[:2], torch.zeros_like(gradient[:2])))
                self.assertGreater(gradient[:4].abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertIsNone(model.net.classifier.bias.grad)
                self.assertTrue(torch.equal(model.net.classifier.weight, weight_before))

    def test_observe_uses_observed_targets_and_e1_abs_insertion_and_update_scores(self):
        current = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.],
                                [3., 2., 1., 4.], [1., 4., 2., 3.]])
        not_aug = current + torch.tensor([1., 0., 0., 0.])
        labels = torch.tensor([2, 3, 2, 3])
        true_labels = torch.tensor([3, 2, 3, 2])
        memory = torch.tensor([[4., 3., 2., 1.]])
        not_aug_memory = torch.tensor([[3., 3., 1., 2.]])
        replay = (torch.tensor([7]), not_aug_memory, memory, torch.tensor([0]))
        arc, arc_calls = _arc_model(task=1, replay=replay, num_classes=6, scoring_scale=64)
        e1, e1_calls = _model('scale_cosine_ce', task=1, replay=replay,
                              num_classes=6, scoring_scale=1)
        for model in (arc, e1):
            model.args.alpha_sample_insertion = 0.75
            model.observe(current, labels, not_aug, epoch=0,
                          true_labels=true_labels, sample_ids=torch.arange(4))
        self.assertEqual(len(arc_calls), 4)
        for index in (0, 2):
            self.assertTrue(torch.equal(arc_calls[index][0], e1_calls[index][0]))
        self.assertTrue(torch.equal(arc.buffer.updated_scores[0], replay[0]))
        self.assertTrue(torch.equal(arc.buffer.updated_scores[1], e1.buffer.updated_scores[1]))
        expected_update = F.cross_entropy(e1_calls[2][0], replay[3], reduction='none')
        torch.testing.assert_close(arc.buffer.updated_scores[1], expected_update, rtol=0, atol=0)
        current_scores = F.cross_entropy(e1_calls[0][0], labels, reduction='none')
        selected = torch.topk(current_scores, 1, largest=False).indices
        inserted = arc.buffer.inserted
        self.assertTrue(torch.equal(inserted['sample_ids'], selected))
        torch.testing.assert_close(inserted['sample_selection_scores'], current_scores[selected],
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(inserted['sample_selection_scores'],
                                    e1.buffer.inserted['sample_selection_scores']))
        raw_current = arc._observe_training_logits(current, present=labels.unique())
        observed_target_logits = arcface_training_logits(raw_current, labels)
        oracle_target_logits = arcface_training_logits(raw_current, true_labels)
        self.assertTrue(torch.equal(arc_calls[1][0], observed_target_logits))
        self.assertFalse(torch.equal(arc_calls[1][0], oracle_target_logits))
        raw_replay = arc._observe_training_logits(memory, replay=True)
        self.assertTrue(torch.equal(arc_calls[3][0], arcface_training_logits(raw_replay, replay[3])))

    def test_true_labels_do_not_change_observe_training_loss_or_gradients(self):
        current = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        labels = torch.tensor([2, 3])
        memory = torch.tensor([[4., 3., 2., 1.]])
        replay = (torch.tensor([7]), memory, memory, torch.tensor([0]))
        outputs = []
        for true_labels in (labels, labels.flip(0)):
            model, calls = _arc_model(task=1, replay=replay, num_classes=6)
            loss = model.observe(current, labels, current, epoch=1, true_labels=true_labels)
            outputs.append((model, calls, loss))
        left, left_calls, left_loss = outputs[0]
        right, right_calls, right_loss = outputs[1]
        self.assertEqual(left_loss, right_loss)
        for index in (1, 3):
            self.assertTrue(torch.equal(left_calls[index][0], right_calls[index][0]))
        for name in ('backbone.weight', 'classifier.weight'):
            self.assertTrue(torch.equal(dict(left.net.named_parameters())[name].grad,
                                        dict(right.net.named_parameters())[name].grad))

    def test_cosine_inference_stays_margin_free_seen_only(self):
        inputs = torch.tensor([[1., 2., 3., 4.]])
        arc, _ = _arc_model(task=1, num_classes=6)
        e1, _ = _model('scale_cosine_ce', task=1, num_classes=6)
        for model in (arc, e1):
            model.args.cosine_inference = 1
            model.eval()
        logits = arc.forward(inputs)
        self.assertEqual(tuple(logits.shape), (1, 4))
        torch.testing.assert_close(logits, e1.forward(inputs), rtol=0, atol=0)
        arc.args.arcface_margin = 0
        torch.testing.assert_close(arc.forward(inputs), logits, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
