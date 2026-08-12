import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.buffer import Buffer
from utils.loss_trace import LossTraceRecorder, assign_dynamic_groups


class LossTraceTests(unittest.TestCase):
    def test_dynamic_groups_are_mutually_exclusive_and_use_top_twenty_percent(self):
        groups = assign_dynamic_groups(
            current_losses=torch.tensor([0.2, 0.8]),
            current_labels=torch.tensor([0, 1]),
            current_true_labels=torch.tensor([0, 0]),
            memory_losses=torch.arange(12, dtype=torch.float32),
            memory_labels=torch.zeros(12, dtype=torch.long),
            memory_true_labels=torch.tensor([0] * 10 + [1, 1]),
            memory_source_task_ids=torch.zeros(12, dtype=torch.long),
            current_task_id=1,
        )

        self.assertEqual(groups['current_new'].tolist(), [True, False])
        self.assertEqual(torch.where(groups['memory_hard_old'])[0].tolist(), [8, 9])
        self.assertEqual(groups['memory_easy_old'].sum().item(), 8)
        self.assertEqual(torch.where(groups['memory_noisy'])[0].tolist(), [10, 11])
        self.assertFalse((groups['memory_hard_old'] & groups['memory_easy_old']).any())

    def test_recorder_writes_raw_samples_and_iteration_curve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                conf_jobnum='12345678-test', loss_trace_flush_every=1, loss_trace_dir=temp_dir,
                seed=0, noise_type='symmetric', noise_rate=0.4,
            )
            recorder = LossTraceRecorder(args, 'seq-cifar10', 'er_ace_aer_abs')
            recorder.record(
                task_id=1, epoch_id=5, epoch_iteration_id=2, task_iteration_id=12,
                num_batches_per_epoch=10, replay_on=True,
                current_losses=torch.tensor([0.2, 0.8]),
                current_labels=torch.tensor([0, 1]), current_true_labels=torch.tensor([0, 0]),
                current_sample_ids=torch.tensor([100, 101]), current_source_task_ids=torch.tensor([1, 1]),
                memory_losses=torch.arange(12, dtype=torch.float32),
                memory_labels=torch.zeros(12, dtype=torch.long),
                memory_true_labels=torch.tensor([0] * 10 + [1, 1]),
                memory_sample_ids=torch.arange(200, 212),
                memory_source_task_ids=torch.zeros(12, dtype=torch.long),
                memory_buffer_slot_ids=torch.arange(12),
            )

            with recorder.sample_path.open(encoding='utf-8') as handle:
                sample_rows = list(csv.DictReader(handle))
            with recorder.curve_path.open(encoding='utf-8') as handle:
                curve_rows = list(csv.DictReader(handle))

            self.assertEqual(len(sample_rows), 14)
            self.assertEqual(len(curve_rows), 1)
            self.assertEqual(curve_rows[0]['hard_old_count'], '2')
            self.assertEqual(curve_rows[0]['easy_old_count'], '8')
            self.assertEqual(curve_rows[0]['noisy_count'], '3')
            self.assertAlmostEqual(float(curve_rows[0]['hard_old_loss']), 8.5)
            self.assertAlmostEqual(float(curve_rows[0]['x_epoch']), 5.2)
            self.assertTrue((Path(temp_dir) / 'seq-cifar10_er_ace_aer_abs_12345678' / 'metadata.json').exists())

    def test_buffer_keeps_trace_metadata_aligned(self):
        buffer = Buffer(4, device='cpu', sample_selection_strategy='reservoir')
        buffer.add_data(
            examples=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            labels=torch.tensor([4, 5]), true_labels=torch.tensor([4, 2]),
            task_labels=torch.tensor([0, 1]), sample_ids=torch.tensor([101, 202]),
        )

        self.assertEqual(buffer.sample_ids.dtype, torch.int64)
        self.assertEqual(buffer.sample_ids[:2].tolist(), [101, 202])
        self.assertEqual(buffer.true_labels[:2].tolist(), [4, 2])
        self.assertEqual(buffer.task_labels[:2].tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
