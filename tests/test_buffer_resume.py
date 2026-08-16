import unittest
from argparse import Namespace
from types import SimpleNamespace

import torch

from models.utils.continual_model import ContinualModel
from utils.buffer import ABSSampling, Buffer, LARSSampling


class _TwoTaskDataset:
    @staticmethod
    def get_offsets():
        return 4, 6


class BufferResumeTests(unittest.TestCase):
    def test_empty_new_safe_buffer_roundtrip(self):
        source = Buffer(buffer_size=4, device='cpu')
        restored = Buffer(buffer_size=4, device='cpu')
        owner = SimpleNamespace(
            buffer=restored,
            args=Namespace(buffer_size=4, inference_only=False),
        )

        ContinualModel.load_buffer(owner, source.serialize())

        self.assertTrue(restored.is_empty())
        self.assertEqual(restored.num_seen_examples, 0)
        self.assertFalse(hasattr(restored, 'examples'))

    def test_safe_serialization_restores_reservoir_count_and_abs_scores(self):
        source = Buffer(
            buffer_size=4,
            device='cpu',
            sample_selection_strategy='abs',
            dataset=_TwoTaskDataset(),
        )
        source.add_data(
            examples=torch.arange(12, dtype=torch.float32).reshape(4, 3),
            labels=torch.tensor([0, 1, 2, 3]),
            sample_selection_scores=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        )
        source.num_seen_examples = 123
        expected_scores = source.sample_selection_fn.importance_scores.clone()

        restored = Buffer(
            buffer_size=4,
            device='cpu',
            sample_selection_strategy='abs',
            dataset=_TwoTaskDataset(),
        )
        owner = SimpleNamespace(
            buffer=restored,
            args=Namespace(buffer_size=4, inference_only=False),
        )
        ContinualModel.load_buffer(owner, source.serialize())

        self.assertEqual(restored.num_seen_examples, 123)
        torch.testing.assert_close(
            restored.sample_selection_fn.importance_scores,
            expected_scores,
        )

    def test_legacy_safe_buffer_is_rejected_for_training_resume(self):
        restored = Buffer(buffer_size=4, device='cpu')
        owner = SimpleNamespace(
            buffer=restored,
            args=Namespace(buffer_size=4, inference_only=False),
        )
        legacy_payload = {
            'examples': torch.zeros(4, 3),
            'labels': torch.arange(4),
        }

        with self.assertRaisesRegex(ValueError, 'num_seen_examples'):
            ContinualModel.load_buffer(owner, legacy_payload)

    def test_legacy_safe_buffer_remains_loadable_for_inference(self):
        restored = Buffer(buffer_size=4, device='cpu')
        owner = SimpleNamespace(
            buffer=restored,
            args=Namespace(buffer_size=4, inference_only=True),
        )
        legacy_payload = {
            'examples': torch.zeros(4, 3),
            'labels': torch.arange(4),
        }

        ContinualModel.load_buffer(owner, legacy_payload)

        self.assertEqual(restored.num_seen_examples, 4)

    def test_lars_normalization_replaces_missing_checkpoint_scores(self):
        sampler = LARSSampling(buffer_size=4, device='cpu')

        all_missing = sampler.normalize_scores(torch.full((4,), -torch.inf))
        partly_rebuilt = sampler.normalize_scores(
            torch.tensor([0.2, -torch.inf, 0.8, -torch.inf]),
        )

        self.assertTrue(torch.isfinite(all_missing).all())
        self.assertTrue(torch.isfinite(partly_rebuilt).all())

    def test_abs_resume_probabilities_are_finite_before_all_slots_are_replayed(self):
        sampler = ABSSampling(
            buffer_size=4,
            device='cpu',
            dataset=_TwoTaskDataset(),
        )
        sampler.importance_scores[0] = 0.5
        past_scores, current_scores = sampler.scale_scores(
            torch.tensor([True, True, False, False]),
        )

        self.assertTrue(torch.isfinite(past_scores).all())
        self.assertTrue(torch.isfinite(current_scores).all())
        self.assertAlmostEqual(float(past_scores.sum()), 1.0)
        self.assertAlmostEqual(float(current_scores.sum()), 1.0)


if __name__ == '__main__':
    unittest.main()
