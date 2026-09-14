"""Pure AER/ABS with the validated Power-Normalized Linear Oracle SAP."""

from __future__ import annotations

from argparse import ArgumentParser

from models.dgc_sap import (
    DgcSap,
    SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION,
    SAP_SKIPPED_INFERENCE,
)
from models.er_ace_aer_abs import ErAceAerAbs


SAP_SKIPPED_FINAL_ONLY = 'SAP_SKIPPED_FINAL_ONLY'


class AerSap(ErAceAerAbs):
    """AER/ABS + Oracle Linear SAP, without DGC/OGC."""

    NAME = 'aer_sap'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = ErAceAerAbs.get_parser(parser)
        group = parser.add_argument_group('Power-Normalized Linear Oracle SAP')
        group.add_argument('--sap_batch_size', type=int, default=32)
        group.add_argument(
            '--sap_oracle_reference', type=int, default=1, choices=[1],
            help='Task 1 uses all current-task oracle-clean samples; later tasks '
                 'use all historical-buffer oracle-clean samples plus an equal, '
                 'class-balanced current-task oracle-clean subset.',
        )
        group.add_argument(
            '--sap_oracle_scale', type=float, default=100.0,
            help='SAP scale coefficient for the final Linear projection.',
        )
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        if args.sap_batch_size <= 0:
            raise ValueError('sap_batch_size must be positive')
        if args.sap_oracle_scale <= 0:
            raise ValueError('sap_oracle_scale must be positive')
        self.sap_history = []

    def _should_store_buffer_metadata(self) -> bool:
        # Oracle cleanliness at the task boundary requires the true label of
        # every retained buffer item; this does not affect AER sampling/training.
        return True

    # Reuse the exact validated dgc-sap oracle path. These methods are
    # independent of DGC/OGC state and only consume the AER network/buffer.
    _normalized_batches = DgcSap._normalized_batches
    _record_sap_event = DgcSap._record_sap_event
    _build_oracle_reference_batches = DgcSap._build_oracle_reference_batches
    _run_oracle_classifier_sap = DgcSap._run_oracle_classifier_sap
    _evaluate_seen_task_accuracies = DgcSap._evaluate_seen_task_accuracies
    _build_oracle_projection = DgcSap._build_oracle_projection
    _sap_importance_from_energy = DgcSap._sap_importance_from_energy

    def _run_task_boundary_sap(self, dataset) -> None:
        start_from = getattr(self.args, 'start_from', None)
        if (
            getattr(self.args, 'loadcheck', None) is not None
            and start_from is not None
            and self.current_task < start_from
        ):
            self._record_sap_event(status=SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION)
            return
        if getattr(self.args, 'inference_only', False):
            self._record_sap_event(status=SAP_SKIPPED_INFERENCE)
            return
        self._run_oracle_classifier_sap(dataset)

    def end_task(self, dataset):
        super().end_task(dataset)
        if self.current_task != int(dataset.N_TASKS) - 1:
            self._record_sap_event(status=SAP_SKIPPED_FINAL_ONLY)
            return
        self._run_task_boundary_sap(dataset)
