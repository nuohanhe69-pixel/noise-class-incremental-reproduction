"""CR-01.1: stable IDs across the real aer_sap data/observe/ABS path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from datasets.utils.continual_dataset import store_masked_loaders
from models.aer_sap import AerSap
from utils.buffer import Buffer
from utils.training import train_single_epoch


class _Images:
    def __init__(self, true_labels):
        self.targets = np.asarray(true_labels, dtype=np.int64)
        self.data = np.stack([
            np.full((32, 32, 3), 20 + sample_id * 20, dtype=np.uint8)
            for sample_id in range(len(true_labels))
        ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.as_tensor(self.data[index]).permute(2, 0, 1).float() / 255
        return image, int(self.targets[index]), image.clone()


class _Setting:
    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES = 100
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10

    def __init__(self, args):
        self.args = args
        self.c_task = -1
        self.test_loaders = []

    def get_offsets(self, task_id=None):
        task_id = self.c_task if task_id is None else task_id
        return task_id * 10, (task_id + 1) * 10


class _Progress:
    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class AerSapBufferSampleIdTests(unittest.TestCase):
    def test_ids_survive_no_trace_training_abs_replacement_and_task1_old_reference(self):
        true_labels = np.array([0, 1, 0, 1, 10, 11, 10, 11])
        observed_labels = true_labels.copy()
        observed_labels[1] = 0  # dirty Task0 sample, still in Task0 split
        args = SimpleNamespace(
            model='aer-sap', enable_loss_trace=0, noise_rate=0.2,
            permute_classes=False, validation=None, validation_mode='current',
            label_perc=1, label_perc_by_class=1, batch_size=2,
            drop_last=False, num_workers=0, seed=0,
            fitting_mode='epochs', debug_mode=False, code_optimization=0,
            device='cpu', use_aer=1, n_epochs=2, minibatch_size=1,
            alpha_sample_insertion=0.0, sample_selection_strategy='abs',
            nowand=True,
        )
        setting = _Setting(args)

        def get_task_loader():
            with patch('datasets.utils.continual_dataset.build_noisy_labels',
                       return_value=observed_labels.copy()):
                return store_masked_loaders(
                    _Images(true_labels), _Images(true_labels), setting,
                )[0]

        task0_loader = get_task_loader()
        self.assertEqual(setting.c_task, 0)
        self.assertEqual(task0_loader.dataset.extra_return_fields,
                         ('true_labels', 'sample_ids'))
        self.assertEqual(task0_loader.dataset.indexes.tolist(), [0, 1, 2, 3])
        # Check the actual DataLoader tuple before the training loop consumes it.
        batch = next(iter(task0_loader))
        for image, observed, clean, sample_id in zip(batch[0], batch[1], batch[3], batch[4]):
            sample_id = int(sample_id)
            self.assertTrue(torch.equal(
                image, torch.as_tensor(task0_loader.dataset.data[
                    np.where(task0_loader.dataset.indexes == sample_id)[0][0]
                ]).permute(2, 0, 1).float() / 255,
            ))
            self.assertEqual(int(observed), int(observed_labels[sample_id]))
            self.assertEqual(int(clean), int(true_labels[sample_id]))

        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 100))
        model.device = torch.device('cpu')
        model.args = args
        model.dataset = setting
        model.num_classes = 100
        model.loss = F.cross_entropy
        model.opt = torch.optim.SGD(model.net.parameters(), lr=0.0)
        model.transform = nn.Identity()
        model.normalization_transform = nn.Identity()
        model.buffer = Buffer(3, device='cpu', dataset=setting,
                              sample_selection_strategy='abs')
        model.loss_trace_recorder = None
        model.seen_so_far = torch.empty(0, dtype=torch.long)
        model._current_task = 0
        model._n_seen_classes = 10
        model._past_epoch = 0
        model._task_iteration = 0
        model._epoch_iteration = 0
        self.assertTrue(model._should_store_buffer_metadata())
        inserted_ids = []
        add_data = model.buffer.add_data

        def record_add_data(**kwargs):
            inserted_ids.extend(kwargs['sample_ids'].tolist())
            return add_data(**kwargs)

        np.random.seed(0)
        with patch.object(model.buffer, 'add_data', side_effect=record_add_data):
            train_single_epoch(model, task0_loader, args, epoch=0,
                               pbar=_Progress(), system_tracker=lambda: None)

        self.assertEqual(sorted(inserted_ids), [0, 1, 2, 3])
        self.assertEqual(model.buffer.num_seen_examples, 4)
        self.assertEqual(len(model.buffer), 3)
        self.assertIn(inserted_ids[-1], model.buffer.sample_ids[:3].tolist())
        self.assertIn(1, model.buffer.sample_ids[:3].tolist())
        self.assertIsNone(model.loss_trace_recorder)
        for slot in range(3):
            sample_id = int(model.buffer.sample_ids[slot])
            self.assertIn(sample_id, range(4))
            expected_image = torch.as_tensor(_Images(true_labels).data[sample_id]) \
                .permute(2, 0, 1).float() / 255
            self.assertTrue(torch.equal(model.buffer.examples[slot], expected_image))
            self.assertEqual(int(model.buffer.labels[slot]), int(observed_labels[sample_id]))
            self.assertEqual(int(model.buffer.true_labels[slot]), int(true_labels[sample_id]))
            self.assertEqual(int(model.buffer.task_labels[slot]), 0)

        task1_loader = get_task_loader()
        self.assertEqual(setting.c_task, 1)
        model._current_task = 1
        model._n_seen_classes = 20
        images, labels, tasks, stats, evidence = model._build_oracle_reference_batches(
            setting, return_task_ids=True, return_evidence=True,
        )
        clean_slots = torch.nonzero(
            model.buffer.labels[:3] == model.buffer.true_labels[:3]
        ).flatten()
        old_start = stats['reference_new_count']
        self.assertEqual(stats['reference_old_count'], len(clean_slots))
        self.assertEqual(evidence['sample_ids'][old_start:].tolist(),
                         model.buffer.sample_ids[clean_slots].tolist())
        self.assertNotIn(1, evidence['sample_ids'][old_start:].tolist())
        self.assertEqual(evidence['observed_labels'][old_start:].tolist(),
                         model.buffer.labels[clean_slots].tolist())
        self.assertEqual(labels[old_start:].tolist(),
                         model.buffer.true_labels[clean_slots].tolist())
        self.assertEqual(tasks[old_start:].tolist(), [0] * len(clean_slots))
        for position, slot in enumerate(clean_slots.tolist(), start=old_start):
            self.assertTrue(torch.equal(images[position], model.buffer.examples[slot]))
        self.assertEqual(task1_loader.dataset.extra_return_fields,
                         ('true_labels', 'sample_ids'))
