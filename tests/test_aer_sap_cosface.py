"""CosFace changes only observed-label targets in cosine training CE."""

import argparse
import unittest

import torch
from torch.nn import functional as F

from models.aer_sap import AerSap, cosface_training_logits
from tests.test_aer_sap_normalized_cosine_ce import _model, _scoped_cosine_logits


def _cos_model(margin=0.35, **kwargs):
    model, calls = _model('cosface_ce', **kwargs)
    model.args.cosface_margin = margin
    return model, calls


class AerSapCosFaceTests(unittest.TestCase):
    def test_parser_keeps_defaults_and_all_eight_experiment_routes(self):
        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        defaults = parser.parse_args(['--buffer_size', '4'])
        self.assertEqual(defaults.training_loss, 'ce')
        self.assertEqual(defaults.cosface_margin, 0.35)
        self.assertEqual(defaults.arcface_margin, 0.5)
        self.assertEqual(defaults.cosine_inference, 0)
        self.assertEqual(defaults.scale_cosine_scoring_scale, 1)
        self.assertEqual(defaults.nce_ace_scope, 'baseline')
        routes = (
            ('ce', []), ('normalized_cosine_ce', []),
            ('scale_cosine_ce', ['--scale_cosine_scoring_scale', '1']),
            ('scale_cosine_ce', ['--scale_cosine_scoring_scale', '64']),
            ('nce', ['--nce_ace_scope', 'baseline']),
            ('nce', ['--nce_ace_scope', 'aligned']),
            ('arcface_ce', ['--arcface_margin', '0.5']),
            ('cosface_ce', ['--cosface_margin', '0.35']),
        )
        for mode, flags in routes:
            parsed = parser.parse_args(['--buffer_size', '4', '--training_loss', mode] + flags)
            self.assertEqual(parsed.training_loss, mode)

    def test_margin_zero_matches_e1_current_replay_logits_loss_and_gradients_exactly(self):
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        for replay, labels in ((False, torch.tensor([2, 3])), (True, torch.tensor([0, 2]))):
            with self.subTest(replay=replay):
                cos, _ = _cos_model(margin=0, task=1, num_classes=6)
                e1, _ = _model('scale_cosine_ce', task=1, num_classes=6, scoring_scale=1)
                kwargs = {'replay': True} if replay else {'present': labels.unique()}
                raw = cos._observe_training_logits(inputs, **kwargs)
                scaled = e1._observe_training_logits(inputs, **kwargs)
                self.assertTrue(torch.equal(cosface_training_logits(raw, labels, 0), scaled))
                cos_loss = cos._compute_training_loss(raw, labels)
                e1_loss = e1._compute_training_loss(scaled, labels)
                self.assertTrue(torch.equal(cos_loss, e1_loss))
                cos_loss.backward()
                e1_loss.backward()
                for name in ('backbone.weight', 'classifier.weight'):
                    cos_parameter = dict(cos.net.named_parameters())[name]
                    e1_parameter = dict(e1.net.named_parameters())[name]
                    self.assertTrue(torch.equal(cos_parameter.grad, e1_parameter.grad), name)
                self.assertIsNone(cos.net.classifier.bias.grad)

    def test_target_only_margin_preserves_input_and_mask_without_clamping(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                masked = torch.finfo(dtype).min
                cosine = torch.tensor([[0.3, 0.6, masked, masked],
                                       [0.9, -0.95, 0.1, masked]],
                                      dtype=dtype, requires_grad=True)
                original = cosine.detach().clone()
                labels = torch.tensor([0, 1])
                logits = cosface_training_logits(cosine, labels)
                expected = original.clone()
                expected[torch.arange(2), labels] -= 0.35
                active = original != masked
                expected[active] *= 64
                self.assertTrue(torch.equal(logits, expected))
                self.assertTrue(torch.equal(logits[~active], original[~active]))
                self.assertTrue(torch.equal(cosine.detach(), original))
                self.assertLess(logits[1, 1].item(), -64)
                non_targets = active.clone()
                non_targets[torch.arange(2), labels] = False
                self.assertTrue(torch.equal(logits[non_targets], original[non_targets] * 64))
                self.assertTrue(torch.isfinite(logits).all())
                F.cross_entropy(logits, labels).backward()
                self.assertTrue(torch.isfinite(cosine.grad).all())
                self.assertTrue(torch.equal(cosine.grad[~active],
                                            torch.zeros_like(cosine.grad[~active])))

    def test_invalid_margin_and_inactive_target_rejected(self):
        cosine = torch.tensor([[0.3, 0.6, torch.finfo(torch.float32).min]])
        for margin in (-0.1, float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                cosface_training_logits(cosine, torch.tensor([0]), margin=margin)
        with self.assertRaisesRegex(ValueError, 'target class must be active'):
            cosface_training_logits(cosine, torch.tensor([2]))

    def test_current_present_only_replay_seen_only_bias_free_and_future_gradients(self):
        inputs = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        for replay, labels in ((False, torch.tensor([2, 3])), (True, torch.tensor([0, 2]))):
            with self.subTest(replay=replay):
                model, _ = _cos_model(task=1, num_classes=6)
                weight_before = model.net.classifier.weight.detach().clone()
                with torch.no_grad():
                    model.net.classifier.bias.fill_(float('nan'))
                kwargs = {'replay': True} if replay else {'present': labels.unique()}
                raw = model._observe_training_logits(inputs, **kwargs)
                expected = _scoped_cosine_logits(
                    model.net, inputs, 1, slice(0, 4) if replay else labels.unique(),
                )
                torch.testing.assert_close(raw, expected, rtol=0, atol=0)
                loss = model._compute_training_loss(raw, labels)
                loss.backward()
                gradient = model.net.classifier.weight.grad
                self.assertTrue(torch.equal(gradient[4:], torch.zeros_like(gradient[4:])))
                if not replay:
                    self.assertTrue(torch.equal(gradient[:2], torch.zeros_like(gradient[:2])))
                self.assertGreater(gradient[:4].abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNone(model.net.classifier.bias.grad)
                self.assertTrue(torch.equal(model.net.classifier.weight, weight_before))

    def test_observed_targets_and_e1_abs_insertion_selected_ids_and_update_scores(self):
        current = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.],
                                [3., 2., 1., 4.], [1., 4., 2., 3.]])
        not_aug = current + torch.tensor([1., 0., 0., 0.])
        labels = torch.tensor([2, 3, 2, 3])
        true_labels = torch.tensor([3, 2, 3, 2])
        memory = torch.tensor([[4., 3., 2., 1.]])
        not_aug_memory = torch.tensor([[3., 3., 1., 2.]])
        replay = (torch.tensor([7]), not_aug_memory, memory, torch.tensor([0]))
        cos, cos_calls = _cos_model(task=1, replay=replay, num_classes=6, scoring_scale=64)
        e1, e1_calls = _model('scale_cosine_ce', task=1, replay=replay,
                              num_classes=6, scoring_scale=1)
        for model in (cos, e1):
            model.args.alpha_sample_insertion = 0.75
            model.observe(current, labels, not_aug, epoch=0,
                          true_labels=true_labels, sample_ids=torch.arange(4))
        self.assertEqual(len(cos_calls), 4)
        for index in (0, 2):
            self.assertTrue(torch.equal(cos_calls[index][0], e1_calls[index][0]))
        self.assertTrue(torch.equal(cos.buffer.updated_scores[0], replay[0]))
        self.assertTrue(torch.equal(cos.buffer.updated_scores[1], e1.buffer.updated_scores[1]))
        expected_update = F.cross_entropy(e1_calls[2][0], replay[3], reduction='none')
        torch.testing.assert_close(cos.buffer.updated_scores[1], expected_update, rtol=0, atol=0)
        current_scores = F.cross_entropy(e1_calls[0][0], labels, reduction='none')
        selected = torch.topk(current_scores, 1, largest=False).indices
        inserted = cos.buffer.inserted
        self.assertTrue(torch.equal(inserted['sample_ids'], selected))
        self.assertTrue(torch.equal(inserted['sample_ids'], e1.buffer.inserted['sample_ids']))
        torch.testing.assert_close(inserted['sample_selection_scores'], current_scores[selected],
                                   rtol=0, atol=0)
        self.assertTrue(torch.equal(inserted['sample_selection_scores'],
                                    e1.buffer.inserted['sample_selection_scores']))
        raw_current = cos._observe_training_logits(current, present=labels.unique())
        observed_target_logits = cosface_training_logits(raw_current, labels)
        oracle_target_logits = cosface_training_logits(raw_current, true_labels)
        self.assertTrue(torch.equal(cos_calls[1][0], observed_target_logits))
        self.assertFalse(torch.equal(cos_calls[1][0], oracle_target_logits))
        raw_replay = cos._observe_training_logits(memory, replay=True)
        self.assertTrue(torch.equal(cos_calls[3][0], cosface_training_logits(raw_replay, replay[3])))

    def test_changing_only_true_labels_preserves_logits_loss_and_gradients(self):
        current = torch.tensor([[1., 2., 3., 4.], [2., 1., 4., 3.]])
        labels = torch.tensor([2, 3])
        memory = torch.tensor([[4., 3., 2., 1.]])
        replay = (torch.tensor([7]), memory, memory, torch.tensor([0]))
        outputs = []
        for true_labels in (labels, labels.flip(0)):
            model, calls = _cos_model(task=1, replay=replay, num_classes=6)
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

    def test_inference_matches_e1_and_arcface_without_margin_or_bias(self):
        inputs = torch.tensor([[1., 2., 3., 4.]])
        cos, _ = _cos_model(task=1, num_classes=6)
        e1, _ = _model('scale_cosine_ce', task=1, num_classes=6)
        arc, _ = _model('arcface_ce', task=1, num_classes=6)
        arc.args.arcface_margin = 0.5
        for model in (cos, e1, arc):
            model.args.cosine_inference = 1
            model.eval()
        logits = cos.forward(inputs)
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertTrue(torch.equal(logits, e1.forward(inputs)))
        self.assertTrue(torch.equal(logits, arc.forward(inputs)))
        cos.args.cosface_margin = 0
        with torch.no_grad():
            cos.net.classifier.bias.fill_(float('nan'))
        self.assertTrue(torch.equal(cos.forward(inputs), logits))


if __name__ == '__main__':
    unittest.main()
