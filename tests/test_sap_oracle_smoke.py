"""Smoke test for the oracle task-boundary SAP end-to-end pipeline.

Builds a synthetic CIFAR-10-like task (with true_labels), instantiates
DgcSap, and runs the oracle boundary function directly. Validates:
- Gram collection hook fires and yields the right shape.
- Projection is applied to the classifier weight (and bias is preserved).
- A SAP_ORACLE_EXECUTED event is recorded with sensible stats.
"""

import unittest

import torch
from torch import nn

from backbone.ResNetBlock import resnet18
from models.dgc_sap import DgcSap, SAP_FAILED, SAP_ORACLE_EXECUTED


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


class OracleEndToEndSmokeTests(unittest.TestCase):
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
        # An empty buffer (this is task 0 boundary; no old samples yet).
        from utils.buffer import Buffer
        model.buffer = Buffer(buffer_size=4, device=torch.device('cpu'),
                              sample_selection_strategy='reservoir')
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
        self.assertFalse(torch.equal(weight_before[:2], weight_after[:2]))
        self.assertTrue(torch.equal(weight_before[2:], weight_after[2:]))
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
        self.assertEqual(event['total_reference_count'], 25)
        self.assertEqual(event['per_class_reference_count'], [12, 13])
        self.assertEqual(event['buffer_total_count'], 0)
        self.assertEqual(event['old_buffer_clean_count'], 0)
        self.assertEqual(event['gram_shape'], [backbone.classifier.in_features] * 2)
        self.assertGreater(event['gram_trace'], 0.0)
        self.assertEqual(event['sap_alpha'], 100.0)
        self.assertEqual(event['n_seen_classes'], 2)
        self.assertEqual(event['total_classifier_rows'], 10)
        self.assertEqual(event['max_future_row_delta'], 0.0)
        self.assertEqual(event['max_bias_delta'], 0.0)
        self.assertEqual(len(event['seen_task_accuracy_comparisons']), 1)
        self.assertIsNotNone(event['seen_average_accuracy_before'])
        self.assertIsNotNone(event['seen_average_accuracy_after'])
        # Weight stats should be sensible.
        self.assertGreater(event['seen_relative_weight_delta'], 0.0)
        self.assertGreater(event['seen_weight_norm_ratio'], 0.0)

        # The AER restore snapshot must be the post-SAP task-boundary model.
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(model.past_model_ckpt[name], value)

    def test_missing_seen_class_aborts_without_modifying_classifier(self):
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

        self.assertTrue(torch.equal(before, backbone.classifier.weight))
        self.assertEqual(model.sap_history[-1]['status'], SAP_FAILED)
        self.assertEqual(model.sap_history[-1]['oracle_stage'], 'class_balanced_gram')
        self.assertIn('missing seen classes: [1]', model.sap_history[-1]['error_message'])


if __name__ == '__main__':
    unittest.main()
