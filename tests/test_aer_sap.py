"""Contract tests for the pure AER + Oracle Linear SAP ablation entry."""

import argparse
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from backbone.ResNetBlock import resnet18
from models import get_model_names
from models.dgc import DGC
from models.dgc_sap import DgcSap, SAP_ORACLE_EXECUTED
from models.er_ace_aer_abs import ErAceAerAbs
from tests.test_sap_oracle_smoke import _FakeTrainDataset
from utils.buffer import Buffer


class AerSapContractTests(unittest.TestCase):
    def test_inherits_aer_directly_without_dgc(self):
        from models.aer_sap import AerSap

        self.assertIs(AerSap.__bases__[0], ErAceAerAbs)
        self.assertNotIn(DGC, inspect.getmro(AerSap))

    def test_parser_has_oracle_sap_but_no_ogc_arguments(self):
        from models.aer_sap import AerSap

        parser = AerSap.get_parser(argparse.ArgumentParser(add_help=False))
        destinations = {action.dest for action in parser._actions}

        self.assertIn('sap_oracle_reference', destinations)
        self.assertIn('sap_oracle_scale', destinations)
        self.assertIn('sap_batch_size', destinations)
        self.assertFalse(any(name.startswith('ogc_') for name in destinations))

    def test_reuses_exact_dgc_sap_oracle_pipeline(self):
        from models.aer_sap import AerSap

        shared_methods = (
            '_normalized_batches',
            '_build_oracle_reference_batches',
            '_run_oracle_classifier_sap',
            '_evaluate_seen_task_accuracies',
            '_build_oracle_projection',
            '_sap_importance_from_energy',
        )
        for method_name in shared_methods:
            self.assertIs(
                getattr(AerSap, method_name),
                getattr(DgcSap, method_name),
                msg=f'{method_name} must stay identical to dgc-sap oracle mode',
            )

    def test_model_registry_recognizes_hyphenated_cli_name(self):
        from models.aer_sap import AerSap

        self.assertIs(get_model_names()['aer-sap'], AerSap)

    def test_enables_oracle_buffer_metadata_without_enabling_loss_trace(self):
        from models.aer_sap import AerSap

        model = AerSap.__new__(AerSap)
        model.loss_trace_recorder = None

        self.assertTrue(model._should_store_buffer_metadata())
        self.assertIsNone(model.loss_trace_recorder)

        labels = torch.tensor([2, 3, 2])
        model._current_task = 4
        source_task_ids = model._buffer_source_task_ids(None, labels)
        torch.testing.assert_close(source_task_ids, torch.tensor([4, 4, 4]))

    def test_oracle_boundary_projects_full_linear_and_refreshes_aer_checkpoint(self):
        from argparse import Namespace
        from models.aer_sap import AerSap

        true_labels = torch.arange(16, dtype=torch.long) % 2
        observed_labels = true_labels.clone()
        observed_labels[::4] = 1 - observed_labels[::4]
        train_dataset = _FakeTrainDataset(16, 2, true_labels, observed_labels)

        class _Loader:
            dataset = train_dataset

        class _Dataset:
            train_loader = _Loader()
            test_loaders = [[(
                torch.rand(8, 3, 32, 32),
                torch.arange(8, dtype=torch.long) % 2,
            )]]

        backbone = resnet18(num_classes=10, num_filters=4)
        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.net = backbone
        model.device = torch.device('cpu')
        model._current_task = 0
        model._n_classes_current_task = 2
        model._n_seen_classes = 2
        model.buffer = Buffer(4, torch.device('cpu'), sample_selection_strategy='reservoir')
        model.sap_history = []
        model.args = Namespace(sap_oracle_scale=100.0, sap_batch_size=4)

        def batches(images, labels=None):
            tensor = torch.as_tensor(images).float() / 255
            if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
            for start in range(0, len(tensor), 4):
                batch = tensor[start:start + 4]
                if labels is None:
                    yield batch
                else:
                    yield batch, torch.as_tensor(labels[start:start + 4]).long()

        model._normalized_batches = batches
        weight_before = backbone.classifier.weight.detach().clone()
        bias_before = backbone.classifier.bias.detach().clone()

        model._run_oracle_classifier_sap(_Dataset())

        self.assertFalse(torch.equal(weight_before, backbone.classifier.weight))
        self.assertFalse(torch.equal(weight_before[2:], backbone.classifier.weight[2:]))
        torch.testing.assert_close(bias_before, backbone.classifier.bias)
        self.assertEqual(model.sap_history[-1]['status'], SAP_ORACLE_EXECUTED)
        self.assertEqual(model.sap_history[-1]['total_reference_count'], 12)
        for name, value in backbone.state_dict().items():
            torch.testing.assert_close(model.past_model_ckpt[name], value)

    def test_only_final_task_runs_sap_while_every_aer_boundary_runs(self):
        from models.aer_sap import AerSap, SAP_SKIPPED_FINAL_ONLY

        model = AerSap.__new__(AerSap)
        nn.Module.__init__(model)
        model.sap_history = []
        dataset = SimpleNamespace(N_TASKS=10)

        with (
            patch.object(ErAceAerAbs, 'end_task', autospec=True) as aer_end_task,
            patch.object(model, '_run_task_boundary_sap') as run_sap,
        ):
            for task_id in range(dataset.N_TASKS):
                model._current_task = task_id
                model.end_task(dataset)

        self.assertEqual(aer_end_task.call_count, dataset.N_TASKS)
        run_sap.assert_called_once_with(dataset)
        self.assertEqual(len(model.sap_history), dataset.N_TASKS - 1)
        self.assertTrue(all(
            event['status'] == SAP_SKIPPED_FINAL_ONLY
            for event in model.sap_history
        ))
        self.assertEqual(
            [event['task_id'] for event in model.sap_history],
            list(range(dataset.N_TASKS - 1)),
        )


if __name__ == '__main__':
    unittest.main()
