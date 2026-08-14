"""Baseline + DGC with task-boundary scaled activation projection."""

from __future__ import annotations

import copy
import logging
from argparse import ArgumentParser
from dataclasses import replace

import torch

from models.dgc import DGC
from utils.augmentations import apply_transform
from utils.sap_reference import (
    SAPReferenceMemory,
    SAPTrajectorySnapshot,
    build_trajectory_snapshot,
    deserialize_reference_memories,
    promote_pending_references,
    score_seen_class_samples,
    select_task_references,
    serialize_reference_memories,
    split_trusted_and_pending,
)
from utils.sap_runtime import (
    diagnose_reference_purity,
    extract_cifar_task_tensors,
    run_sap_projection_transaction,
    validate_reference_gate,
)


SAP_DRY_RUN = 'SAP_DRY_RUN'
SAP_FAILED = 'SAP_FAILED'
SAP_REJECTED_SAFETY_GATE = 'SAP_REJECTED_SAFETY_GATE'
SAP_SKIPPED_INFERENCE = 'SAP_SKIPPED_INFERENCE'
SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION = 'SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION'


class DgcSap(DGC):
    NAME = 'dgc_sap'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = DGC.get_parser(parser)
        group = parser.add_argument_group('Scaled Activation Projection (SAP)')
        group.add_argument('--sap_reference_per_class', type=int, default=15)
        group.add_argument('--sap_scale', type=float, default=30000.0)
        group.add_argument('--sap_max_activation_patches', type=int, default=20000)
        group.add_argument('--sap_batch_size', type=int, default=32)
        group.add_argument('--sap_dry_run', type=int, default=0, choices=[0, 1])
        group.add_argument('--sap_score_epochs', type=int, nargs=3, default=[35, 45, 50])
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        if args.sap_reference_per_class <= 0:
            raise ValueError('sap_reference_per_class must be positive')
        if args.sap_scale <= 0:
            raise ValueError('sap_scale must be positive')
        if args.sap_max_activation_patches <= 0:
            raise ValueError('sap_max_activation_patches must be positive')
        if args.sap_batch_size <= 0:
            raise ValueError('sap_batch_size must be positive')
        if sorted(set(args.sap_score_epochs)) != list(args.sap_score_epochs):
            raise ValueError('sap_score_epochs must contain three increasing unique epochs')
        self.sap_reference_memory = SAPReferenceMemory()
        self.sap_pending_memory = SAPReferenceMemory()
        self.sap_state_migrated_from_v1 = False
        self.sap_trajectory_history = {}
        self.sap_history = []

    def _normalized_batches(self, images, labels=None):
        for start in range(0, len(images), self.args.sap_batch_size):
            batch_images = images[start:start + self.args.sap_batch_size].to(self.device)
            if batch_images.dtype == torch.uint8:
                batch_images = batch_images.float().div(255)
            else:
                batch_images = batch_images.float()
            normalized = apply_transform(batch_images, self.normalization_transform)
            if labels is None:
                yield normalized
            else:
                yield normalized, labels[start:start + self.args.sap_batch_size].to(self.device)

    def _reference_batch_factory(self):
        references = self.sap_reference_memory.items()
        images = torch.stack([reference.image for reference in references])

        def factory():
            return self._normalized_batches(images)

        return factory

    def _reference_labeled_batch_factory(self):
        references = self.sap_reference_memory.items()
        images = torch.stack([reference.image for reference in references])
        labels = torch.tensor([reference.observed_label for reference in references])

        def factory():
            return self._normalized_batches(images, labels)

        return factory

    def _buffer_labeled_batch_factory(self):
        if self.buffer.is_empty():
            return None
        buffer_data = self.buffer.get_all_data(device='cpu')
        images, labels = buffer_data[0], buffer_data[1]

        def factory():
            return self._normalized_batches(images, labels)

        return factory

    def _buffer_diagnostic_factories(self):
        aggregate = self._buffer_labeled_batch_factory()
        if aggregate is None:
            return {}
        buffer_data = self.buffer.get_all_data(device='cpu')
        images, labels = buffer_data[0], buffer_data[1]
        factories = {'replay_buffer': aggregate}
        task_ids = torch.div(labels, self.n_classes_current_task, rounding_mode='floor')
        for task_id in task_ids.unique(sorted=True).tolist():
            mask = task_ids == task_id
            task_images = images[mask]
            task_labels = labels[mask]

            def factory(images=task_images, labels=task_labels):
                return self._normalized_batches(images, labels)

            factories[f'replay_task_{int(task_id)}'] = factory
        return factories

    def _test_task_batch_factories(self, dataset):
        factories = []
        for test_loader in dataset.test_loaders:
            def factory(loader=test_loader):
                for data in loader:
                    yield data[0].to(self.device), data[1].to(self.device)
            factories.append(factory)
        return factories

    def _record_sap_event(self, **event) -> None:
        event = {'task_id': int(self.current_task), **event}
        self.sap_history.append(event)
        logging.info('SAP event: %s', event)

    def _record_current_task_trajectory(self, dataset, epoch_number: int) -> None:
        images, observed_labels, sample_ids = extract_cifar_task_tensors(
            dataset.train_loader.dataset
        )
        scores = score_seen_class_samples(
            self.net,
            self._normalized_batches(images, observed_labels),
            seen_classes=self.n_seen_classes,
        )
        snapshot = build_trajectory_snapshot(
            epoch=epoch_number,
            sample_ids=sample_ids,
            observed_labels=observed_labels,
            scores=scores,
            ogc_probability_threshold=float(self.ogc_loss_fn.prob_thr_current),
            ogc_low_conf_weight=float(self.args.ogc_low_conf_weight),
            ogc_buffer_penalty_coeff=float(self.args.ogc_buffer_penalty_coeff),
            alpha_sample_insertion=float(self.args.alpha_sample_insertion),
        )
        self.sap_trajectory_history.setdefault(int(self.current_task), []).append(snapshot)
        logging.info(
            'SAP trajectory: task=%d epoch=%d samples=%d pred_agreement=%.6f '
            'ogc_high=%.6f abs_eligible=%.6f',
            self.current_task,
            epoch_number,
            len(snapshot.sample_ids),
            (snapshot.predictions == snapshot.observed_labels).float().mean().item(),
            snapshot.ogc_high_confidence.float().mean().item(),
            snapshot.abs_insertion_eligible.float().mean().item(),
        )

    def end_epoch(self, epoch, dataset):
        super().end_epoch(epoch, dataset)
        epoch_number = int(epoch) + 1
        if epoch_number in self.args.sap_score_epochs:
            self._record_current_task_trajectory(dataset, epoch_number)

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

        task_dataset = dataset.train_loader.dataset
        images, observed_labels, sample_ids = extract_cifar_task_tensors(task_dataset)
        scores = score_seen_class_samples(
            self.net,
            self._normalized_batches(images, observed_labels),
            seen_classes=self.n_seen_classes,
        )
        references, reports = select_task_references(
            images=images,
            observed_labels=observed_labels,
            sample_ids=sample_ids,
            source_task_id=self.current_task,
            losses=scores.losses,
            predictions=scores.predictions,
            confidences=scores.confidences,
            class_quota=self.args.sap_reference_per_class,
            seed=int(self.args.seed or 0) + self.current_task * 1000,
        )
        trusted_references, pending_references = split_trusted_and_pending(references)
        trajectory_snapshots = self.sap_trajectory_history.get(int(self.current_task), [])
        promotion_reports = []
        promoted_references = []
        required_epochs = tuple(int(epoch) for epoch in self.args.sap_score_epochs)
        if pending_references and {snapshot.epoch for snapshot in trajectory_snapshots} == set(required_epochs):
            promoted_references, pending_references, promotion_reports = promote_pending_references(
                pending_references,
                trajectory_snapshots,
                promoted_at_task_id=self.current_task,
                required_epochs=required_epochs,
            )
            trusted_references.extend(promoted_references)
        elif pending_references:
            promotion_reports = [{
                'sample_id': reference.sample_id,
                'observed_label': reference.observed_label,
                'promoted': False,
                'snapshots': [],
                'rejection_reasons': ('MISSING_REQUIRED_SNAPSHOTS',),
            } for reference in pending_references]
        self.sap_reference_memory.add_task(self.current_task, trusted_references)
        self.sap_pending_memory.add_task(self.current_task, pending_references)
        true_labels = getattr(task_dataset, 'true_labels', None)
        diagnostic_reference_purity = diagnose_reference_purity(references, true_labels)
        current_class_count = self.n_classes_current_task
        promoted_by_label = {}
        for reference in promoted_references:
            promoted_by_label[reference.observed_label] = promoted_by_label.get(reference.observed_label, 0) + 1
        gate_reports = [replace(
            report,
            selected_main_count=(
                report.selected_main_count + promoted_by_label.get(report.observed_label, 0)
            ),
        ) for report in reports]
        gate = validate_reference_gate(
            self.sap_reference_memory,
            gate_reports,
            current_class_count=current_class_count,
            seen_class_count=self.n_seen_classes,
        )
        report_state = [report.__dict__.copy() for report in reports]
        if not gate.should_execute:
            self._record_sap_event(
                status=gate.status,
                gate=gate.__dict__.copy(),
                class_reports=report_state,
                diagnostic_reference_purity=diagnostic_reference_purity,
                trusted_added=len(trusted_references),
                pending_added=len(pending_references),
                promoted_added=len(promoted_references),
                promotion_reports=promotion_reports,
            )
            return

        try:
            diagnostic_factories = {
                'reference': self._reference_labeled_batch_factory(),
            }
            diagnostic_factories.update(self._buffer_diagnostic_factories())
            transaction = run_sap_projection_transaction(
                self.net,
                self._reference_batch_factory(),
                total_images=len(self.sap_reference_memory),
                max_patches=self.args.sap_max_activation_patches,
                scale=self.args.sap_scale,
                seed=int(self.args.seed or 0) + self.current_task * 10000,
                dry_run=bool(self.args.sap_dry_run),
                diagnostic_batch_factories=diagnostic_factories,
                seen_classes=self.n_seen_classes,
                test_task_batch_factories=self._test_task_batch_factories(dataset),
                enforce_candidate_safety=True,
            )
            if transaction.committed:
                self.past_model_ckpt = copy.deepcopy(self.net.state_dict())
            if self.args.sap_dry_run:
                status = SAP_DRY_RUN
            elif transaction.accepted:
                status = gate.status
            else:
                status = SAP_REJECTED_SAFETY_GATE
            self._record_sap_event(
                status=status,
                gate=gate.__dict__.copy(),
                class_reports=report_state,
                diagnostic_reference_purity=diagnostic_reference_purity,
                trusted_added=len(trusted_references),
                pending_added=len(pending_references),
                promoted_added=len(promoted_references),
                promotion_reports=promotion_reports,
                candidate_accepted=transaction.accepted,
                candidate_rejection_reasons=transaction.rejection_reasons,
                minimum_weight_norm_ratio=transaction.minimum_weight_norm_ratio,
                training_side_accuracy_deltas=transaction.training_side_accuracy_deltas,
                layer_stats={name: stats.__dict__.copy() for name, stats in transaction.layer_stats.items()},
                comparisons={
                    name: {
                        **comparison.__dict__,
                        'loss_tertiles': {
                            group: group_stats.__dict__.copy()
                            for group, group_stats in comparison.loss_tertiles.items()
                        },
                    }
                    for name, comparison in transaction.comparisons.items()
                },
                max_non_target_state_delta=transaction.max_non_target_state_delta,
                task_accuracy_comparisons=[
                    comparison.__dict__.copy()
                    for comparison in transaction.task_accuracy_comparisons
                ],
            )
        except Exception as error:
            self._record_sap_event(
                status=SAP_FAILED,
                error_type=type(error).__name__,
                error_message=str(error),
                gate=gate.__dict__.copy(),
                class_reports=report_state,
                diagnostic_reference_purity=diagnostic_reference_purity,
                trusted_added=len(trusted_references),
                pending_added=len(pending_references),
                promoted_added=len(promoted_references),
                promotion_reports=promotion_reports,
            )
            logging.exception('SAP failed at task boundary; original model weights were preserved.')

    def end_task(self, dataset):
        super().end_task(dataset)
        self._run_task_boundary_sap(dataset)

    def serialize_sap_state(self) -> dict:
        return {
            'version': 2,
            'reference_memories': serialize_reference_memories(
                self.sap_reference_memory,
                self.sap_pending_memory,
            ),
            'migrated_from_v1': self.sap_state_migrated_from_v1,
            'trajectory_history': {
                str(task_id): [snapshot.__dict__.copy() for snapshot in snapshots]
                for task_id, snapshots in sorted(self.sap_trajectory_history.items())
            },
            'history': copy.deepcopy(self.sap_history),
        }

    def load_sap_state(self, state: dict) -> None:
        version = state.get('version')
        if version == 1:
            memory_state = state['reference_memory']
        elif version == 2:
            memory_state = state['reference_memories']
        else:
            raise ValueError('unsupported dgc-sap checkpoint state')
        (
            self.sap_reference_memory,
            self.sap_pending_memory,
            migrated,
        ) = deserialize_reference_memories(memory_state)
        self.sap_state_migrated_from_v1 = bool(state.get('migrated_from_v1', False) or migrated)
        self.sap_trajectory_history = {
            int(task_id): [SAPTrajectorySnapshot(**snapshot) for snapshot in snapshots]
            for task_id, snapshots in state.get('trajectory_history', {}).items()
        }
        self.sap_history = copy.deepcopy(state.get('history', []))
