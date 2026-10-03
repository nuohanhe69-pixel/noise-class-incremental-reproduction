import io
import random
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import torch

from models.utils.continual_model import ContinualModel
from utils.buffer import (
    ABSSampling, BUFFER_CHECKPOINT_STATE_KEY, BUFFER_CHECKPOINT_STATE_VERSION,
    Buffer, LARSSampling,
)


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

    @staticmethod
    def _abs_buffer():
        return Buffer(buffer_size=4, device='cpu',
                      sample_selection_strategy='abs', dataset=_TwoTaskDataset())

    @staticmethod
    def _checkpoint_roundtrip(source):
        # Exercise an actual safe checkpoint disk representation (in memory).
        checkpoint = io.BytesIO()
        torch.save({'buffer': source.serialize()}, checkpoint)
        checkpoint.seek(0)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restored = BufferResumeTests._abs_buffer()
        owner = SimpleNamespace(buffer=restored,
                                args=Namespace(buffer_size=4, inference_only=False))
        ContinualModel.load_buffer(owner, payload['buffer'])
        return restored

    def test_partial_abs_resume_fills_unused_slots_without_probabilities(self):
        source = self._abs_buffer()
        source.add_data(examples=torch.arange(6, dtype=torch.float32).reshape(2, 3),
                        labels=torch.tensor([0, 4]),
                        sample_selection_scores=torch.tensor([-0.1, -0.2]))
        scores = source.sample_selection_fn.importance_scores.clone()
        self.assertTrue(torch.isfinite(scores[:2]).all())
        self.assertTrue(torch.isneginf(scores[2:]).all())

        restored = self._checkpoint_roundtrip(source)
        self.assertEqual(restored.num_seen_examples, 2)
        torch.testing.assert_close(restored.sample_selection_fn.importance_scores,
                                   scores, rtol=0, atol=0)
        torch.testing.assert_close(restored.examples, source.examples, rtol=0, atol=0)
        torch.testing.assert_close(restored.labels, source.labels, rtol=0, atol=0)
        selector = restored.sample_selection_fn
        call = ABSSampling.__call__
        indices = []

        def record(sampler, *args, **kwargs):
            index = call(sampler, *args, **kwargs)
            indices.append(int(index))
            return index

        with patch.object(ABSSampling, '__call__', record), \
                patch.object(selector, 'scale_scores', side_effect=AssertionError(
                    'Partial buffers must not calculate replacement probabilities')):
            for slot in (2, 3):
                restored.add_data(examples=torch.full((1, 3), float(slot)),
                                  labels=torch.tensor([slot]),
                                  sample_selection_scores=torch.tensor([-0.1 * (slot + 1)]))
                self.assertEqual(restored.num_seen_examples, slot + 1)
                self.assertEqual(indices[-1], slot)
                torch.testing.assert_close(restored.examples[slot],
                                           torch.full((3,), float(slot)))
                if slot == 2:
                    self.assertTrue(torch.isneginf(selector.importance_scores[3]))
        self.assertEqual(indices, [2, 3])
        self.assertTrue(torch.isfinite(selector.importance_scores).all())
        torch.testing.assert_close(selector.importance_scores[:2], scores[:2],
                                   rtol=0, atol=0)

    def test_full_abs_resume_matches_uninterrupted_continuation(self):
        # ABS uses NumPy's global RNG; also restore Python/Torch state explicitly.
        initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        try:
            for seed in (0, 17, 103):
                with self.subTest(seed=seed):
                    source = self._abs_buffer()
                    source.add_data(
                        examples=torch.arange(12, dtype=torch.float32).reshape(4, 3),
                        labels=torch.tensor([0, 1, 4, 5]),
                        sample_selection_scores=torch.tensor([-0.1, -0.2, -0.3, -0.4]))
                    source.num_seen_examples = 9
                    restored = self._checkpoint_roundtrip(source)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
                    candidates = torch.arange(120, dtype=torch.float32).reshape(40, 3) + 100
                    labels = torch.tensor([0, 4, 1, 5] * 10)
                    scores = -torch.arange(1, 41, dtype=torch.float32) / 43

                    def continuation(buffer):
                        indices, states = [], []
                        call = ABSSampling.__call__

                        def record(sampler, *args, **kwargs):
                            index = call(sampler, *args, **kwargs)
                            indices.append(int(np.asarray(index).item()))
                            return index

                        with patch.object(ABSSampling, '__call__', record):
                            for i in range(len(candidates)):
                                buffer.add_data(examples=candidates[i:i + 1],
                                                labels=labels[i:i + 1],
                                                sample_selection_scores=scores[i:i + 1])
                                states.append(buffer.serialize())
                                # serialize() clones selector scores, but buffer
                                # attributes need snapshots for per-step comparison.
                                states[-1] = {
                                    key: value.clone() if isinstance(value, torch.Tensor) else value
                                    for key, value in states[-1].items()}
                        return indices, states

                    uninterrupted_indices, uninterrupted_states = continuation(source)
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2])
                    resumed_indices, resumed_states = continuation(restored)
                    self.assertEqual(uninterrupted_indices, resumed_indices)
                    self.assertGreaterEqual(sum(i >= 0 for i in resumed_indices), 2)
                    self.assertIn(-1, resumed_indices)
                    self.assertEqual(source.num_seen_examples, 49)
                    self.assertEqual(restored.num_seen_examples, 49)
                    for left, right in zip(uninterrupted_states, resumed_states):
                        self.assertEqual(left.keys(), right.keys())
                        for key in left:
                            if key != BUFFER_CHECKPOINT_STATE_KEY:
                                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                        left_state = left[BUFFER_CHECKPOINT_STATE_KEY]
                        right_state = right[BUFFER_CHECKPOINT_STATE_KEY]
                        for field in ('version', 'num_seen_examples', 'sample_selection_strategy'):
                            self.assertEqual(left_state[field], right_state[field])
                        self.assertEqual(left_state['sample_selection_state'].keys(),
                                         right_state['sample_selection_state'].keys())
                        for field in left_state['sample_selection_state']:
                            torch.testing.assert_close(
                                left_state['sample_selection_state'][field],
                                right_state['sample_selection_state'][field], rtol=0, atol=0)
        finally:
            random.setstate(initial_rng[0])
            np.random.set_state(initial_rng[1])
            torch.set_rng_state(initial_rng[2])

    def test_lars_normalization_matches_master_fixed_inputs(self):
        sampler = LARSSampling(buffer_size=4, device='cpu')
        values = torch.tensor([-2., 0., 4., 6.], dtype=torch.float64)
        expected = torch.tensor([0., 2., 6., 8.], dtype=torch.float64) / (8 + 1e-9)
        torch.testing.assert_close(sampler.normalize_scores(values), expected, rtol=0, atol=0)
        constant = torch.tensor([0.25, 0.25])
        torch.testing.assert_close(sampler.normalize_scores(constant), constant, rtol=0, atol=0)
        self.assertIsNone(sampler.normalize_scores(torch.empty(0)))

    def test_lars_normalization_does_not_repair_incomplete_checkpoint_state(self):
        sampler = LARSSampling(buffer_size=4, device='cpu')
        # Master leaves invalid full-buffer scores invalid; the loader owns resume state.
        for values in (torch.full((4,), -torch.inf),
                       torch.tensor([0.2, -torch.inf, 0.8, -torch.inf])):
            with self.subTest(values=values):
                self.assertTrue(torch.isnan(sampler.normalize_scores(values)).all())

    def test_abs_scale_scores_matches_master_past_and_current(self):
        sampler = ABSSampling(buffer_size=6, device='cpu', dataset=_TwoTaskDataset())
        sampler.importance_scores = torch.tensor([2., 4., 6., 1., 3., 5.], dtype=torch.float64)
        past, current = sampler.scale_scores(torch.tensor([True, True, True, False, False, False]))
        normalized = torch.tensor([0., 2., 4.], dtype=torch.float64) / (4 + 1e-9)
        expected_past = 1 - normalized
        expected_past /= expected_past.sum()
        expected_current = normalized / normalized.sum()
        torch.testing.assert_close(past, expected_past, rtol=0, atol=0)
        torch.testing.assert_close(current, expected_current, rtol=0, atol=0)

    def test_abs_past_denominator_has_no_epsilon(self):
        sampler = ABSSampling(buffer_size=4, device='cpu', dataset=_TwoTaskDataset())
        sampler.importance_scores = torch.tensor([1 - 5e-10, 1 - 5e-10, 0., 0.],
                                                 dtype=torch.float64)
        past, current = sampler.scale_scores(torch.tensor([True, True, False, False]))
        expected = torch.tensor([0.5, 0.5], dtype=torch.float64)
        torch.testing.assert_close(past, expected, rtol=0, atol=0)
        torch.testing.assert_close(current, expected, rtol=0, atol=0)

    def test_checkpoint_version_and_strategy_validation_is_preserved(self):
        for field, value, message in (
                ('version', BUFFER_CHECKPOINT_STATE_VERSION + 1, 'version'),
                ('sample_selection_strategy', 'lars', 'strategy mismatch'),
                ('num_seen_examples', -1, 'num_seen_examples')):
            with self.subTest(field=field):
                source = self._abs_buffer()
                payload = source.serialize()
                payload[BUFFER_CHECKPOINT_STATE_KEY][field] = value
                restored = self._abs_buffer()
                owner = SimpleNamespace(buffer=restored,
                                        args=Namespace(buffer_size=4, inference_only=False))
                with self.assertRaisesRegex(ValueError, message):
                    ContinualModel.load_buffer(owner, payload)


if __name__ == '__main__':
    unittest.main()
