"""Smoke test for the oracle task-boundary SAP end-to-end pipeline.

Builds a synthetic CIFAR-10-like task (with true_labels), instantiates
DgcSap, and runs the oracle boundary function directly. Validates:
- Gram collection hook fires and yields the right shape.
- Projection is applied to the classifier weight (and bias is preserved).
- A SAP_ORACLE_EXECUTED event is recorded with sensible stats.
"""

import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch import nn

from backbone.ResNetBlock import resnet18
from models.dgc_sap import DgcSap, SAP_ORACLE_EXECUTED


class _FakeTrainDataset:
    """Mimics the Mammoth task dataset surface used by extract_cifar_task_tensors."""

    def __init__(self, num_samples: int, num_classes: int, true_labels, observed_labels):
        import numpy as _np
        # NHWC uint8 (N, H, W, C) — the format extract_cifar_task_tensors expects.
        rng = torch.Generator().manual_seed(0)
        self.data = torch.randint(0, 256, (num_samples, 32, 32, 3), dtype=torch.uint8, generator=rng).numpy()
        self.targets = list(observed_labels.tolist())
        self.indexes = _np.arange(num_samples)
        self.true_labels = true_labels.tolist()


class _FakeBuffer:
    def __init__(self, images, observed_labels, true_labels, source_task_ids):
        self.images = images
        self.observed_labels = observed_labels
        self.true_labels = true_labels
        self.task_labels = source_task_ids

    def is_empty(self):
        return len(self.images) == 0

    def get_all_data(self, device='cpu'):
        return self.images.to(device), self.observed_labels.to(device)


def _reference_model(train_dataset, buffer, current_task, seed=0):
    class _Loader:
        dataset = train_dataset

    class _Dataset:
        train_loader = _Loader()

    model = DgcSap.__new__(DgcSap)
    nn.Module.__init__(model)
    model.buffer = buffer
    model.dataset = _Dataset()
    model._current_task = current_task
    model._n_classes_current_task = len(set(train_dataset.true_labels))
    model.args = Namespace(seed=seed)
    return model, model.dataset


class OracleEndToEndSmokeTests(unittest.TestCase):
    def test_task1_reference_evidence_matches_actual_selection_without_resampling(self):
        true_labels = torch.tensor([10] * 6 + [11] * 6)
        observed = true_labels.clone()
        observed[0] = 11
        train = _FakeTrainDataset(12, 2, true_labels, observed)
        buffer = _FakeBuffer(
            torch.arange(6 * 3 * 32 * 32, dtype=torch.int32).remainder(256)
            .to(torch.uint8).reshape(6, 3, 32, 32),
            torch.tensor([0, 1, 2, 0, 1, 2]),
            torch.tensor([0, 1, 3, 0, 1, 2]),
            torch.tensor([0, 0, 0, 1, 1, 0]),
        )
        buffer.sample_ids = torch.tensor([40, 41, 42, 43, 44, 45])
        model, dataset = _reference_model(train, buffer, current_task=1, seed=0)
        with patch('models.dgc_sap.torch.randperm', wraps=torch.randperm) as randperm:
            images, labels, tasks, stats, evidence = model._build_oracle_reference_batches(
                dataset, return_task_ids=True, return_evidence=True,
            )
        self.assertEqual(randperm.call_count, 2)
        self.assertEqual(stats['reference_new_count'], 3)
        self.assertEqual(stats['reference_old_count'], 3)
        self.assertEqual(evidence['counts']['old_candidate'], 4)
        self.assertEqual(evidence['counts']['old_eligible'], 3)
        self.assertEqual(evidence['counts']['new_eligible'], 11)
        self.assertEqual(evidence['sample_ids'][3:].tolist(), [40, 41, 45])
        self.assertEqual(evidence['observed_labels'][3:].tolist(), [0, 1, 2])
        self.assertTrue(torch.equal(evidence['true_labels'], labels))
        self.assertTrue(torch.equal(evidence['source_task_ids'], tasks))
        self.assertEqual(evidence['reference_order'].tolist(), list(range(6)))
        for position in range(3):
            sample_id = evidence['sample_ids'][position]
            expected_image = torch.as_tensor(train.data[sample_id]).permute(2, 0, 1)
            self.assertTrue(torch.equal(images[position], expected_image))
        for position, buffer_index in enumerate((0, 1, 5), start=3):
            self.assertTrue(torch.equal(images[position], buffer.images[buffer_index]))
        second = model._build_oracle_reference_batches(
            dataset, return_task_ids=True, return_evidence=True,
        )
        self.assertTrue(torch.equal(second[4]['sample_ids'], evidence['sample_ids']))

    def test_oracle_boundary_projects_classifier_and_records_event(self):
        torch.manual_seed(0)
        # Build a small 2-class task with 32 samples, 25 clean references.
        true_labels = torch.arange(32, dtype=torch.long) % 2
        # Corrupt 20% of the observed labels.
        observed = true_labels.clone()
        observed[::5] = 1 - observed[::5]

        fake_train = _FakeTrainDataset(
            num_samples=32, num_classes=2,
            true_labels=true_labels, observed_labels=observed,
        )

        class _FakeLoader:
            def __init__(self, dataset):
                self.dataset = dataset

        class _FakeDataset:
            N_CLASSES_PER_TASK = 1

            def __init__(self, train_loader):
                self.train_loader = train_loader
                test_inputs = torch.rand(8, 3, 32, 32)
                test_labels = torch.arange(8) % 2
                self.test_loaders = [[(test_inputs, test_labels)]]

        fake_dataset = _FakeDataset(_FakeLoader(fake_train))

        backbone = resnet18(num_classes=10, num_filters=4)

        # Build a minimal Namespace mimicking the relevant args.
        from argparse import Namespace
        args = Namespace(
            sap_oracle_reference=1,
            sap_oracle_scale=100.0,
            sap_batch_size=8,
            sap_score_epochs=[35, 45, 50],
            noise_rate=0.2,
            noise_type='symm',
            seed=0,
            debug_mode=1,
        )
        # DgcSap's parent classes need a loss callable. Use plain CE.
        loss = nn.CrossEntropyLoss()
        # transform not actually used by the oracle path.
        model = DgcSap.__new__(DgcSap)
        nn.Module.__init__(model)
        # Minimal attribute setup needed for the oracle method.
        # These are @property on ContinualModel, so set the private backing fields.
        model.net = backbone
        model.device = torch.device('cpu')
        model.opt = torch.optim.SGD(backbone.parameters(), lr=0.0)
        model._current_task = 0
        model._n_classes_current_task = 1
        model._n_seen_classes = 2
        # Task 1 must report real buffer diagnostics while excluding the
        # buffer from its reference set.
        model.buffer = _FakeBuffer(
            torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8),
            torch.tensor([4, 5, 6]),
            torch.tensor([4, 5, 6]),
            torch.tensor([0, 0, 0]),
        )
        model.dataset = fake_dataset
        model.normalization_transform = lambda x: x  # identity in this smoke test
        model.ogc_loss_fn = None
        model.loss_trace_recorder = None
        model.current_batch_idx = 0
        model.sap_history = []
        model.args = args
        # Compatibility shim: extract_cifar_task_tensors calls .data etc.
        # which our fake already provides.

        # Patch _normalized_batches since it requires a real transform pair.
        def _fake_normalized_batches(images, labels=None):
            import numpy as _np
            arr = torch.as_tensor(_np.asarray(images))
            if arr.dtype == torch.uint8:
                arr = arr.float().div_(255)
            else:
                arr = arr.float()
            # Convert NHWC->NCHW (this is what real pipeline does after normalization).
            if arr.ndim == 4 and arr.shape[-1] in (1, 3):
                arr = arr.permute(0, 3, 1, 2).contiguous()
            bs = 8
            for i in range(0, len(arr), bs):
                if labels is None:
                    yield arr[i:i + bs]
                else:
                    yield arr[i:i + bs], torch.as_tensor(labels[i:i + bs]).long()

        model._normalized_batches = _fake_normalized_batches

        # Snapshot weight and bias before the oracle projection.
        weight_before = backbone.classifier.weight.detach().clone()
        bias_before = backbone.classifier.bias.detach().clone()

        # Run the oracle pipeline.
        model._run_oracle_classifier_sap(fake_dataset)

        # The classifier weight must have changed (projection is non-identity).
        weight_after = backbone.classifier.weight.detach()
        self.assertFalse(torch.equal(weight_before, weight_after))
        self.assertFalse(torch.equal(weight_before[2:], weight_after[2:]))
        # Bias must be untouched (oracle mode projects only the input side).
        bias_after = backbone.classifier.bias.detach()
        torch.testing.assert_close(bias_after, bias_before)

        # An SAP_ORACLE_EXECUTED event must be recorded with the expected stats.
        self.assertEqual(len(model.sap_history), 1)
        event = model.sap_history[0]
        self.assertEqual(event['status'], SAP_ORACLE_EXECUTED)
        self.assertEqual(event['projection_target'], 'classifier')
        self.assertEqual(event['projection_location'], 'pre')
        # All 32 task samples are clean after the 80% filter (we flipped 7 of 32).
        self.assertEqual(event['current_task_clean_count'], 25)
        self.assertEqual(event['current_task_clean_total'], 25)
        self.assertEqual(event['current_task_clean_selected'], 25)
        self.assertEqual(event['historical_buffer_clean_count'], 0)
        self.assertEqual(event['reference_new_count'], 25)
        self.assertEqual(event['reference_old_count'], 0)
        self.assertEqual(event['current_class_selected_counts'], {0: 12, 1: 13})
        self.assertEqual(event['reference_sampling_seed'], 0)
        self.assertEqual(event['total_reference_count'], 25)
        self.assertEqual(event['buffer_total_count'], 3)
        self.assertEqual(event['buffer_clean_count'], 3)
        self.assertEqual(event['gram_shape'], [backbone.classifier.in_features] * 2)
        self.assertGreater(event['gram_trace'], 0.0)
        for statistic in ('min', 'median', 'mean', 'max'):
            self.assertGreater(event[f'feature_norm_before_{statistic}'], 0.0)
            self.assertAlmostEqual(
                event[f'feature_norm_after_{statistic}'], 1.0, places=5,
            )
        self.assertEqual(event['sap_alpha'], 100.0)
        self.assertEqual(event['n_seen_classes'], 2)
        self.assertEqual(event['total_classifier_rows'], 10)
        self.assertEqual(event['max_bias_delta'], 0.0)
        self.assertEqual(len(event['seen_task_accuracy_comparisons']), 1)
        self.assertIsNotNone(event['seen_average_accuracy_before'])
        self.assertIsNotNone(event['seen_average_accuracy_after'])
        # Weight stats should be sensible.
        self.assertGreater(event['relative_weight_delta'], 0.0)
        self.assertGreater(event['weight_norm_ratio'], 0.0)
        self.assertLessEqual(event['importance_min'], event['importance_median'])
        self.assertLessEqual(event['importance_median'], event['importance_max'])

        # The AER restore snapshot must be the post-SAP task-boundary model.
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(model.past_model_ckpt[name], value)

    def test_reference_set_is_not_rejected_when_a_seen_class_is_absent(self):
        true_labels = torch.zeros(8, dtype=torch.long)
        fake_train = _FakeTrainDataset(8, 2, true_labels, true_labels.clone())

        class _Loader:
            dataset = fake_train

        class _Dataset:
            N_CLASSES_PER_TASK = 2
            train_loader = _Loader()
            test_loaders = []

        from argparse import Namespace
        from utils.buffer import Buffer
        backbone = resnet18(num_classes=10, num_filters=4)
        model = DgcSap.__new__(DgcSap)
        nn.Module.__init__(model)
        model.net = backbone
        model.device = torch.device('cpu')
        model._current_task = 0
        model._n_classes_current_task = 2
        model._n_seen_classes = 2
        model.buffer = Buffer(4, torch.device('cpu'), sample_selection_strategy='reservoir')
        model.dataset = _Dataset()
        model.sap_history = []
        model.args = Namespace(sap_oracle_scale=100.0, sap_batch_size=4)

        def _batches(images, labels=None):
            tensor = torch.as_tensor(images).float() / 255
            if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
            for start in range(0, len(tensor), 4):
                yield tensor[start:start + 4], labels[start:start + 4]

        model._normalized_batches = _batches
        before = backbone.classifier.weight.detach().clone()

        model._run_oracle_classifier_sap(model.dataset)

        self.assertFalse(torch.equal(before, backbone.classifier.weight))
        self.assertEqual(model.sap_history[-1]['status'], SAP_ORACLE_EXECUTED)
        self.assertEqual(model.sap_history[-1]['total_reference_count'], 8)

    def test_task1_ignores_buffer_and_keeps_all_current_clean_samples(self):
        true_labels = torch.tensor([0, 0, 1, 1])
        observed_labels = torch.tensor([0, 1, 1, 1])
        train = _FakeTrainDataset(4, 2, true_labels, observed_labels)
        buffer = _FakeBuffer(
            torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8),
            torch.tensor([4, 5, 6]),
            torch.tensor([4, 5, 6]),
            torch.tensor([0, 0, 0]),
        )
        model, dataset = _reference_model(train, buffer, current_task=0)

        images, labels, stats = model._build_oracle_reference_batches(dataset)

        self.assertEqual(len(images), 3)
        self.assertEqual(labels.tolist(), [0, 1, 1])
        self.assertEqual(stats['current_task_clean_total'], 3)
        self.assertEqual(stats['current_task_clean_count'], 3)
        self.assertEqual(stats['current_task_clean_selected'], 3)
        self.assertEqual(stats['buffer_total_count'], 3)
        self.assertEqual(stats['buffer_clean_count'], 3)
        self.assertEqual(stats['reference_new_count'], 3)
        self.assertEqual(stats['reference_old_count'], 0)
        self.assertEqual(stats['historical_buffer_clean_count'], 0)

    def test_later_tasks_balance_new_old_and_select_history_by_source_task(self):
        true_labels = torch.tensor([2] * 8 + [3] * 8)
        train = _FakeTrainDataset(16, 2, true_labels, true_labels.clone())
        buffer_true = torch.tensor([0, 0, 1, 1, 1, 2, 3, 0])
        buffer_observed = torch.tensor([0, 0, 1, 1, 1, 2, 3, 9])
        buffer = _FakeBuffer(
            torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8),
            buffer_observed,
            buffer_true,
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 0]),
        )
        model, dataset = _reference_model(train, buffer, current_task=1, seed=17)

        images, labels, stats = model._build_oracle_reference_batches(dataset)

        self.assertEqual(stats['reference_new_count'], 5)
        self.assertEqual(stats['reference_old_count'], 5)
        self.assertGreaterEqual(
            stats['current_task_clean_count'], stats['current_task_clean_selected'],
        )
        self.assertGreaterEqual(
            stats['buffer_clean_count'], stats['historical_buffer_clean_count'],
        )
        self.assertEqual(stats['buffer_clean_count'], 7)
        self.assertEqual(stats['historical_buffer_clean_count'], 5)
        self.assertEqual(
            stats['total_reference_count'],
            stats['reference_new_count'] + stats['reference_old_count'],
        )
        self.assertEqual(stats['total_reference_count'], 10)
        self.assertEqual(len(images), 10)
        self.assertEqual(labels[5:].tolist(), [0, 0, 1, 1, 1])
        self.assertNotIn(2, labels[5:].tolist())
        self.assertNotIn(3, labels[5:].tolist())

    def test_optional_task_ids_are_aligned_without_resampling_reference(self):
        true_labels = torch.tensor([2] * 8 + [3] * 8)
        train = _FakeTrainDataset(16, 2, true_labels, true_labels.clone())
        buffer_true = torch.tensor([0, 0, 1, 1, 1, 2, 3, 0])
        buffer_observed = torch.tensor([0, 0, 1, 1, 1, 2, 3, 9])
        buffer = _FakeBuffer(
            torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8),
            buffer_observed,
            buffer_true,
            torch.tensor([0, 0, 0, 0, 0, 1, 1, 0]),
        )
        model, dataset = _reference_model(train, buffer, current_task=1, seed=17)

        with patch('models.dgc_sap.torch.randperm', wraps=torch.randperm) as randperm:
            images, labels, task_ids, stats = model._build_oracle_reference_batches(
                dataset, return_task_ids=True,
            )

        self.assertEqual(randperm.call_count, 2)
        self.assertEqual(len(images), len(labels))
        self.assertEqual(len(labels), len(task_ids))
        self.assertEqual(task_ids[:stats['reference_new_count']].tolist(), [1] * 5)
        self.assertEqual(task_ids[stats['reference_new_count']:].tolist(), [0] * 5)
        self.assertEqual(labels[stats['reference_new_count']:].tolist(), [0, 0, 1, 1, 1])

    def test_reference_builder_default_return_contract_is_unchanged(self):
        true_labels = torch.tensor([0, 0, 1, 1])
        train = _FakeTrainDataset(4, 2, true_labels, true_labels.clone())
        buffer = _FakeBuffer(
            torch.empty(0, 3, 32, 32, dtype=torch.uint8),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
        model, dataset = _reference_model(train, buffer, current_task=0)

        result = model._build_oracle_reference_batches(dataset)

        self.assertEqual(len(result), 3)

    def test_current_class_sampling_is_balanced_and_remainder_goes_to_first_classes(self):
        true_labels = torch.tensor([20] * 8 + [21] * 8 + [22] * 8)
        train = _FakeTrainDataset(24, 3, true_labels, true_labels.clone())
        old_count = 8
        buffer = _FakeBuffer(
            torch.randint(0, 256, (old_count, 3, 32, 32), dtype=torch.uint8),
            torch.arange(old_count) % 2,
            torch.arange(old_count) % 2,
            torch.zeros(old_count, dtype=torch.long),
        )
        model, dataset = _reference_model(train, buffer, current_task=2, seed=0)

        _, _, stats = model._build_oracle_reference_batches(dataset)

        selected_counts = stats['current_class_selected_counts']
        self.assertLessEqual(max(selected_counts.values()) - min(selected_counts.values()), 1)
        self.assertEqual(selected_counts, {20: 3, 21: 3, 22: 2})
        self.assertEqual(stats['reference_sampling_seed'], 2)

    def test_sampling_is_reproducible_for_same_seed_and_task(self):
        true_labels = torch.tensor([2] * 10 + [3] * 10)
        train = _FakeTrainDataset(20, 2, true_labels, true_labels.clone())
        buffer = _FakeBuffer(
            torch.randint(0, 256, (7, 3, 32, 32), dtype=torch.uint8),
            torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            torch.zeros(7, dtype=torch.long),
        )
        model_a, dataset_a = _reference_model(train, buffer, current_task=1, seed=9)
        model_b, dataset_b = _reference_model(train, buffer, current_task=1, seed=9)

        images_a, labels_a, stats_a = model_a._build_oracle_reference_batches(dataset_a)
        torch.rand(1000)  # perturb global RNG; local sampling must remain unchanged
        images_b, labels_b, stats_b = model_b._build_oracle_reference_batches(dataset_b)

        torch.testing.assert_close(images_a, images_b)
        torch.testing.assert_close(labels_a, labels_b)
        self.assertEqual(stats_a, stats_b)

    def test_noisy_foreign_true_label_does_not_hide_historical_buffer_samples(self):
        true_labels = torch.tensor([2, 2, 3, 3, 0])
        observed_labels = torch.tensor([2, 2, 3, 3, 2])
        train = _FakeTrainDataset(5, 3, true_labels, observed_labels)
        buffer = _FakeBuffer(
            torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
        )
        model, dataset = _reference_model(train, buffer, current_task=1, seed=0)

        _, labels, stats = model._build_oracle_reference_batches(dataset)

        self.assertEqual(stats['reference_new_count'], 2)
        self.assertEqual(stats['reference_old_count'], 2)
        self.assertEqual(labels[-2:].tolist(), [0, 1])
        self.assertEqual(set(stats['current_class_selected_counts']), {2, 3})

    def test_oracle_dgc_requests_stable_source_task_metadata(self):
        model = DgcSap.__new__(DgcSap)
        nn.Module.__init__(model)
        model.args = Namespace(sap_oracle_reference=1)
        model.loss_trace_recorder = None
        model._current_task = 3

        self.assertTrue(model._should_store_buffer_metadata())
        torch.testing.assert_close(
            model._buffer_source_task_ids(None, torch.tensor([4, 5])),
            torch.tensor([3, 3]),
        )


if __name__ == '__main__':
    unittest.main()
