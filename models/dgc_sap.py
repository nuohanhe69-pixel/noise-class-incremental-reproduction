"""Baseline + DGC with task-boundary scaled activation projection."""

from __future__ import annotations

import copy
import logging
from argparse import ArgumentParser
from dataclasses import replace

import torch

from models.dgc import DGC
from utils.augmentations import apply_transform
from utils.sap import (
    build_sap_projection_from_gram,
    collect_classifier_input_features,
    project_linear_weight,
)
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
SAP_ORACLE_EXECUTED = 'SAP_ORACLE_EXECUTED'


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
        oracle_group = parser.add_argument_group('Oracle task-boundary SAP')
        oracle_group.add_argument('--sap_oracle_reference', type=int, default=0,
                                choices=[0, 1],
                                help='Use the oracle (clean + full-task) reference set '
                                     'and project only the final classifier (Linear) layer. '
                                     'Bypasses Robust GMM / promotion / safety gates.')
        oracle_group.add_argument('--sap_oracle_scale', type=float, default=100.0,
                                help='SAP scale coefficient in oracle mode. Defaults to 100 '
                                     'because the 512-dim classifier Gram concentrates energy '
                                     'much more than the late-stage conv patch Grams, so the '
                                     'conv default of 30000 would behave near-identity.')
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
        if args.sap_oracle_scale <= 0:
            raise ValueError('sap_oracle_scale must be positive')
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
        # In oracle mode we do not use the trajectory-snapshot promotion
        # pipeline; skip the per-epoch full-task scoring pass entirely to
        # save time.
        if self.args.sap_oracle_reference:
            return
        epoch_number = int(epoch) + 1
        if epoch_number in self.args.sap_score_epochs:
            self._record_current_task_trajectory(dataset, epoch_number)

    def _build_oracle_reference_batches(self, dataset):
        """Build the list of (images, true_labels) batches for the oracle ref set.

        Current task: every training image, but only those whose observed
        label equals the true label (oracle cleanliness). For a CIFAR-10 task
        with symm-20% noise this gives ~8000 samples (10,000 × (1-noise_rate)).

        Buffer: only samples whose ``task_id != current_task`` AND whose
        observed label equals the true label. ``buffer.task_labels`` may be
        absent when loss-trace is disabled; in that case we infer the task id
        via ``labels // n_classes_per_task`` (same fallback as
        ``_buffer_diagnostic_factories``).

        Labels are returned purely for diagnostics; the projection itself
        only consumes images.
        """
        task_dataset = dataset.train_loader.dataset
        images, observed_labels, sample_ids = extract_cifar_task_tensors(task_dataset)
        true_labels = getattr(task_dataset, 'true_labels', None)
        if true_labels is None:
            raise ValueError('oracle mode requires the task dataset to expose true_labels')

        true_labels_cpu = torch.as_tensor(true_labels, dtype=torch.long).reshape(-1)
        observed_cpu = observed_labels.detach().cpu().long()
        sample_ids_cpu = sample_ids.detach().cpu().long()
        clean_mask = observed_cpu == true_labels_cpu[sample_ids_cpu]
        task_images = images[clean_mask]
        task_true_labels = true_labels_cpu[sample_ids_cpu][clean_mask]
        task_count = int(task_images.shape[0])

        buffer_images = None
        buffer_true = None
        buffer_total = 0
        buffer_old_count = 0

        if not self.buffer.is_empty():
            buf = self.buffer.get_all_data(device='cpu')
            buf_images, buf_labels = buf[0], buf[1]
            if buf_images is not None and buf_labels is not None and buf_images.numel():
                if hasattr(self.buffer, 'true_labels') and self.buffer.true_labels is not None:
                    buf_true = self.buffer.true_labels.detach().cpu().long()
                else:
                    raise ValueError('oracle mode requires buffer.true_labels to be set')

                if hasattr(self.buffer, 'task_labels') and self.buffer.task_labels is not None:
                    buf_task_id = self.buffer.task_labels.detach().cpu().long()
                else:
                    n_per_task = getattr(self.dataset, 'N_CLASSES_PER_TASK', None) \
                        or max(1, self.n_classes_current_task)
                    buf_task_id = (buf_labels.detach().cpu().long() // n_per_task)

                buf_labels_cpu = buf_labels.detach().cpu().long()
                clean = buf_labels_cpu == buf_true
                old = buf_task_id != int(self.current_task)
                keep = clean & old
                buffer_images = buf_images[keep]
                buffer_true = buf_true[keep]
                buffer_total = int(buf_images.shape[0])
                buffer_old_count = int(buffer_images.shape[0])

        if buffer_images is not None and buffer_old_count > 0:
            all_images = torch.cat([task_images, buffer_images], dim=0)
            all_true_labels = torch.cat([task_true_labels, buffer_true], dim=0)
        else:
            all_images = task_images
            all_true_labels = task_true_labels

        return all_images, all_true_labels, task_count, buffer_total, buffer_old_count

    def _run_oracle_classifier_sap(self, dataset) -> None:
        """Oracle task-boundary SAP: project only the final classifier Linear.

        Procedure:

        1. Build the oracle reference set (current task clean + buffer old clean).
        2. Stream the classifier-input Gram matrix via the pre-hook.
        3. Build the input-side SAP projection matrix ``Mr`` (no truncation).
        4. Project the classifier weight in place; bias is untouched.
        5. Update ``past_model_ckpt`` so AER restores the projected weights at
           the next task's fitting epoch (mirrors the existing committed path).
        6. Log a SAP_ORACLE_EXECUTED event with stats.
        """
        try:
            all_images, all_true_labels, task_count, buffer_total, buffer_old_count = \
                self._build_oracle_reference_batches(dataset)
        except Exception as error:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='reference_construction',
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception('Oracle SAP: reference construction failed.')
            return

        total_images = int(all_images.shape[0])
        if total_images == 0:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='empty_reference_set',
            )
            return

        # Build batches of normalized inputs.
        batches_with_labels = list(self._normalized_batches(all_images, all_true_labels))

        def image_batches():
            for images, _labels in batches_with_labels:
                yield images

        try:
            features = collect_classifier_input_features(
                self.net,
                image_batches(),
                total_images=total_images,
            )
        except Exception as error:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='feature_collection',
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception('Oracle SAP: classifier-input feature collection failed.')
            return

        # Build the streaming Gram from the captured features.
        gram = features.transpose(0, 1) @ features
        if not torch.isfinite(gram).all():
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='non_finite_gram',
            )
            return

        try:
            projection, spectrum = self._build_oracle_projection(gram)
        except Exception as error:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='projection_build',
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception('Oracle SAP: projection matrix construction failed.')
            return

        # Locate the classifier module and project its weight in place.
        classifier = self.net.classifier if hasattr(self.net, 'classifier') \
            else self.net.get_submodule('fc')
        if not isinstance(classifier, torch.nn.Linear):
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='classifier_not_found',
            )
            return

        weight_before = classifier.weight.detach().clone()
        try:
            projected_weight, weight_stats = project_linear_weight(weight_before, projection)
        except Exception as error:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='weight_projection',
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception('Oracle SAP: linear weight projection failed.')
            return

        with torch.no_grad():
            classifier.weight.copy_(projected_weight.to(classifier.weight.dtype))

        # Mirror the existing committed SAP behaviour: refresh the AER snapshot
        # so the next task's fitting epoch restores the projected weights.
        self.past_model_ckpt = copy.deepcopy(self.net.state_dict())

        # Free-form accuracy diagnostics on the reference set using the
        # captured features (no extra forward pass needed).
        ref_accuracy_before = None
        ref_accuracy_after = None
        try:
            ref_labels = torch.cat(
                [_labels for _b_images, _labels in batches_with_labels], dim=0
            )
            with torch.no_grad():
                logits_before = features.to(weight_before.device) @ weight_before.transpose(0, 1) \
                    + classifier.bias.detach()
                logits_after = features.to(projected_weight.device) @ projected_weight.transpose(0, 1) \
                    + classifier.bias.detach()
                ref_labels = ref_labels.to(logits_before.device)
                ref_accuracy_before = (
                    (logits_before.argmax(dim=1) == ref_labels).float().mean().item()
                )
                ref_accuracy_after = (
                    (logits_after.argmax(dim=1) == ref_labels).float().mean().item()
                )
        except Exception:
            logging.exception('Oracle SAP: reference accuracy diagnostics failed.')

        spectrum_np = spectrum.detach().cpu().numpy() if isinstance(spectrum, torch.Tensor) \
            else spectrum
        self._record_sap_event(
            status=SAP_ORACLE_EXECUTED,
            projection_location='pre',
            projection_target='classifier',
            current_task_clean_count=task_count,
            buffer_total_count=buffer_total,
            buffer_old_clean_count=buffer_old_count,
            total_reference_images=total_images,
            projection_min_eigenvalue=float(spectrum_np.min()),
            projection_max_eigenvalue=float(spectrum_np.max()),
            projection_trace=float(spectrum_np.sum()),
            projection_effective_rank=int((spectrum_np > 1e-6).sum()),
            reference_accuracy_before=ref_accuracy_before,
            reference_accuracy_after=ref_accuracy_after,
            relative_weight_delta=weight_stats['relative_weight_delta'],
            weight_norm_ratio=weight_stats['weight_norm_ratio'],
        )

    def _build_oracle_projection(self, gram: torch.Tensor):
        """Build the SAP input-side projection Mr from the classifier-input Gram.

        Equivalent to the official SAP path with no basis truncation and the
        scaled importance ``α·sval_ratio/((α−1)·sval_ratio+1)``. Returns both
        the projection matrix and the eigenvalue spectrum (for logging).
        """
        device = gram.device
        if device.type == 'mps':
            gram_for_decomp = gram.cpu()
        else:
            gram_for_decomp = gram

        symmetric_gram = (gram_for_decomp + gram_for_decomp.transpose(0, 1)) * 0.5
        if torch.trace(symmetric_gram) <= 0:
            raise ValueError('oracle Gram has zero trace; cannot build projection')

        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_gram)
        energy = eigenvalues.clamp_min(0)
        if device.type == 'mps':
            energy = energy.to(device)
            eigenvectors = eigenvectors.to(device)
        importance = self._sap_importance_from_energy(energy)
        projection = (eigenvectors * importance.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
        projection = (projection + projection.transpose(0, 1)) * 0.5
        return projection, energy

    def _sap_importance_from_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """Official SAP importance scaled to [0,1] over normalized energies.

        importance = α·r / ((α−1)·r + 1),  r = energy / energy.sum().
        Mirrors `` ``_sap_importance`` `` in `` ``utils/sap.py`` for the conv path.
        """
        total = energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
        ratios = energy / total
        alpha = float(self.args.sap_oracle_scale)
        return alpha * ratios / ((alpha - 1.0) * ratios + 1.0)

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

        if self.args.sap_oracle_reference:
            self._run_oracle_classifier_sap(dataset)
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
                required_replay_task_names={
                    f'replay_task_{task_id}'
                    for task_id in range(self.current_task + 1)
                },
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
