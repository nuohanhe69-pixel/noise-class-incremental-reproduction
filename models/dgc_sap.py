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
    collect_classifier_input_features,
    normalize_classifier_input_features,
    project_linear_weight,
    resolve_classifier_module,
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

    def _should_store_buffer_metadata(self) -> bool:
        return bool(getattr(self.args, 'sap_oracle_reference', 0)) or super()._should_store_buffer_metadata()

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

    def _build_oracle_reference_batches(self, dataset, *, return_task_ids=False):
        """Build the balanced oracle reference set used by Linear SAP.

        Task 1 keeps every oracle-clean current-task sample and ignores the
        buffer. Later tasks keep every historical oracle-clean buffer sample
        as ``Old`` and sample the same number of oracle-clean current-task
        samples as ``New``, balanced across the sorted current classes.
        """
        task_dataset = dataset.train_loader.dataset
        images, observed_labels, sample_ids = extract_cifar_task_tensors(task_dataset)
        true_labels = getattr(task_dataset, 'true_labels', None)
        if true_labels is None:
            raise ValueError('oracle mode requires the task dataset to expose true_labels')

        true_labels_cpu = torch.as_tensor(true_labels, dtype=torch.long).reshape(-1)
        observed_cpu = observed_labels.detach().cpu().long()
        sample_ids_cpu = sample_ids.detach().cpu().long()
        task_true_labels_all = true_labels_cpu[sample_ids_cpu]
        clean_mask = observed_cpu == task_true_labels_all
        clean_task_images = images[clean_mask]
        clean_task_labels = task_true_labels_all[clean_mask]
        current_classes = clean_task_labels.unique(sorted=True)
        current_task_clean_total = int(clean_task_images.shape[0])
        sampling_seed = int(getattr(self.args, 'seed', 0) or 0) + int(self.current_task)

        buffer_images = None
        buffer_true = None
        buffer_task_ids = None
        buffer_total = 0
        buffer_clean_count = 0
        historical_buffer_clean_count = 0

        if not self.buffer.is_empty():
            buf = self.buffer.get_all_data(device='cpu')
            buf_images, buf_labels = buf[0], buf[1]
            if buf_images is not None and buf_labels is not None and buf_images.numel():
                if hasattr(self.buffer, 'true_labels') and self.buffer.true_labels is not None:
                    buf_true = self.buffer.true_labels[:len(buf_labels)].detach().cpu().long()
                else:
                    raise ValueError('oracle mode requires buffer.true_labels to be set')

                buf_labels_cpu = buf_labels.detach().cpu().long()
                clean = buf_labels_cpu == buf_true
                buffer_total = int(buf_images.shape[0])
                buffer_clean_count = int(clean.sum().item())

                # Task 1 records the real buffer diagnostics but never uses
                # buffer samples as references because no historical task exists.
                if self.current_task > 0:
                    if not hasattr(self.buffer, 'task_labels') or self.buffer.task_labels is None:
                        raise ValueError('oracle mode requires buffer.task_labels to be set')
                    buf_source_task_ids = self.buffer.task_labels[:len(buf_labels)].detach().cpu().long()
                    historical = buf_source_task_ids < int(self.current_task)
                    keep = clean & historical
                    buffer_images = buf_images[keep]
                    buffer_true = buf_true[keep]
                    buffer_task_ids = buf_source_task_ids[keep]
                    historical_buffer_clean_count = int(buffer_images.shape[0])

        if self.current_task == 0:
            selected_task_images = clean_task_images
            selected_task_labels = clean_task_labels
        else:
            new_target = historical_buffer_clean_count
            class_count = int(current_classes.numel())
            if class_count == 0 and new_target > 0:
                raise ValueError('oracle mode found no clean current-task classes')

            base = new_target // class_count if class_count else 0
            remainder = new_target % class_count if class_count else 0
            generator = torch.Generator(device='cpu').manual_seed(sampling_seed)
            class_indices_by_label = {
                class_label: torch.nonzero(
                    clean_task_labels == class_label, as_tuple=False,
                ).flatten()
                for class_label in current_classes.tolist()
            }
            if sum(len(indices) for indices in class_indices_by_label.values()) < new_target:
                raise ValueError(
                    f'oracle mode needs {new_target} clean current-task samples, '
                    f'but only {current_task_clean_total} are available'
                )
            class_targets = {
                class_label: min(
                    base + int(class_position < remainder),
                    len(class_indices_by_label[class_label]),
                )
                for class_position, class_label in enumerate(current_classes.tolist())
            }
            deficit = new_target - sum(class_targets.values())
            while deficit:
                eligible = [
                    class_label for class_label in current_classes.tolist()
                    if class_targets[class_label] < len(class_indices_by_label[class_label])
                ]
                class_label = min(eligible, key=lambda label: (class_targets[label], label))
                class_targets[class_label] += 1
                deficit -= 1

            selected_indices = []
            for class_label in current_classes.tolist():
                class_indices = class_indices_by_label[class_label]
                class_target = class_targets[class_label]
                permutation = torch.randperm(len(class_indices), generator=generator)
                selected_indices.append(class_indices[permutation[:class_target]])

            selected_indices = (
                torch.cat(selected_indices)
                if selected_indices else torch.empty(0, dtype=torch.long)
            )
            selected_task_images = clean_task_images[selected_indices]
            selected_task_labels = clean_task_labels[selected_indices]

        current_class_selected_counts = {
            int(class_label): int((selected_task_labels == class_label).sum().item())
            for class_label in current_classes.tolist()
        }
        reference_new_count = int(selected_task_images.shape[0])
        reference_old_count = historical_buffer_clean_count
        selected_task_ids = torch.full(
            (reference_new_count,), int(self.current_task), dtype=torch.long,
        )

        if reference_old_count > 0:
            all_images = torch.cat([selected_task_images, buffer_images], dim=0)
            all_true_labels = torch.cat([selected_task_labels, buffer_true], dim=0)
            all_task_ids = torch.cat([selected_task_ids, buffer_task_ids], dim=0)
        else:
            all_images = selected_task_images
            all_true_labels = selected_task_labels
            all_task_ids = selected_task_ids

        stats = {
            'current_task_clean_total': current_task_clean_total,
            'current_task_clean_count': current_task_clean_total,
            'current_task_clean_selected': reference_new_count,
            'buffer_clean_count': buffer_clean_count,
            'historical_buffer_clean_count': historical_buffer_clean_count,
            'reference_new_count': reference_new_count,
            'reference_old_count': reference_old_count,
            'total_reference_count': int(all_images.shape[0]),
            'current_class_selected_counts': current_class_selected_counts,
            'reference_sampling_seed': sampling_seed,
            'buffer_total_count': buffer_total,
        }
        if return_task_ids:
            return all_images, all_true_labels, all_task_ids, stats
        return all_images, all_true_labels, stats

    def _run_oracle_classifier_sap(self, dataset) -> None:
        """Oracle task-boundary SAP: project only the final classifier Linear.

        Procedure:

        1. Build the oracle reference set: all current-task clean samples on
           Task 1; historical-buffer clean samples plus an equal, class-balanced
           current-task clean subset on later tasks.
        2. Stream the classifier-input Gram matrix via the pre-hook.
        3. Build the input-side SAP projection matrix ``Mr`` (no truncation).
        4. Project the classifier weight in place; bias is untouched.
        5. Update ``past_model_ckpt`` so AER restores the projected weights at
           the next task's fitting epoch (mirrors the existing committed path).
        6. Log a SAP_ORACLE_EXECUTED event with stats.
        """
        try:
            all_images, all_true_labels, reference_stats = \
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

        batches_with_labels = list(self._normalized_batches(all_images, all_true_labels))

        def image_batches():
            for images, _labels in batches_with_labels:
                yield images

        try:
            features = collect_classifier_input_features(
                self.net, image_batches(), total_images=total_images,
            )
            normalized_features, feature_norm_stats = \
                normalize_classifier_input_features(features)
        except Exception as error:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='feature_collection',
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception('Oracle SAP: classifier-input feature collection failed.')
            return

        gram = normalized_features.transpose(0, 1) @ normalized_features
        if not torch.isfinite(gram).all():
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='non_finite_gram',
            )
            return

        try:
            projection, energy, normalized_energy, importance = \
                self._build_oracle_projection(gram)
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
        try:
            classifier = resolve_classifier_module(self.net)
        except ValueError:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='classifier_not_found',
            )
            return

        weight_before = classifier.weight.detach().clone()
        bias_before = (
            classifier.bias.detach().clone() if classifier.bias is not None else None
        )
        try:
            accuracy_before = self._evaluate_seen_task_accuracies(dataset)
        except Exception as error:
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='seen_accuracy_before',
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception('Oracle SAP: pre-projection seen-task evaluation failed.')
            return
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

        try:
            with torch.no_grad():
                classifier.weight.copy_(projected_weight.to(classifier.weight.dtype))
            accuracy_after = self._evaluate_seen_task_accuracies(dataset)
        except Exception as error:
            with torch.no_grad():
                classifier.weight.copy_(weight_before)
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage='seen_accuracy_after',
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception(
                'Oracle SAP: post-projection seen-task evaluation failed; weight restored.'
            )
            return

        # Mirror the existing committed SAP behaviour: refresh the AER snapshot
        # so the next task's fitting epoch restores the projected weights.
        self.past_model_ckpt = copy.deepcopy(self.net.state_dict())

        bias_after = classifier.bias.detach() if classifier.bias is not None else None
        max_bias_delta = (
            0.0 if bias_before is None
            else (bias_after - bias_before).abs().max().item()
        )
        task_accuracy_comparisons = [
            {
                'task_id': before['task_id'],
                'sample_count': before['sample_count'],
                'accuracy_before': before['accuracy'],
                'accuracy_after': after['accuracy'],
                'accuracy_delta': after['accuracy'] - before['accuracy'],
            }
            for before, after in zip(accuracy_before, accuracy_after)
        ]
        seen_average_before = (
            sum(item['accuracy'] for item in accuracy_before) / len(accuracy_before)
            if accuracy_before else None
        )
        seen_average_after = (
            sum(item['accuracy'] for item in accuracy_after) / len(accuracy_after)
            if accuracy_after else None
        )
        self._record_sap_event(
            status=SAP_ORACLE_EXECUTED,
            projection_location='pre',
            projection_target='classifier',
            total_reference_count=total_images,
            current_task_clean_count=reference_stats['current_task_clean_count'],
            buffer_clean_count=reference_stats['buffer_clean_count'],
            current_task_clean_total=reference_stats['current_task_clean_total'],
            current_task_clean_selected=reference_stats['current_task_clean_selected'],
            historical_buffer_clean_count=reference_stats['historical_buffer_clean_count'],
            reference_new_count=reference_stats['reference_new_count'],
            reference_old_count=reference_stats['reference_old_count'],
            current_class_selected_counts=reference_stats['current_class_selected_counts'],
            reference_sampling_seed=reference_stats['reference_sampling_seed'],
            buffer_total_count=reference_stats['buffer_total_count'],
            gram_shape=list(gram.shape),
            gram_trace=gram.trace().item(),
            feature_norm_before_min=feature_norm_stats['before']['min'],
            feature_norm_before_median=feature_norm_stats['before']['median'],
            feature_norm_before_mean=feature_norm_stats['before']['mean'],
            feature_norm_before_max=feature_norm_stats['before']['max'],
            feature_norm_after_min=feature_norm_stats['after']['min'],
            feature_norm_after_median=feature_norm_stats['after']['median'],
            feature_norm_after_mean=feature_norm_stats['after']['mean'],
            feature_norm_after_max=feature_norm_stats['after']['max'],
            gram_eigenvalue_min=energy.min().item(),
            gram_eigenvalue_max=energy.max().item(),
            normalized_energy_min=normalized_energy.min().item(),
            normalized_energy_max=normalized_energy.max().item(),
            normalized_energy_median=normalized_energy.median().item(),
            sap_alpha=float(self.args.sap_oracle_scale),
            importance_min=importance.min().item(),
            importance_max=importance.max().item(),
            importance_median=importance.median().item(),
            importance_trace=importance.sum().item(),
            n_seen_classes=int(self.n_seen_classes),
            total_classifier_rows=int(classifier.weight.shape[0]),
            relative_weight_delta=weight_stats['relative_weight_delta'],
            weight_norm_ratio=weight_stats['weight_norm_ratio'],
            max_bias_delta=max_bias_delta,
            seen_task_accuracy_comparisons=task_accuracy_comparisons,
            seen_average_accuracy_before=seen_average_before,
            seen_average_accuracy_after=seen_average_after,
        )

    def _evaluate_seen_task_accuracies(self, dataset) -> list[dict]:
        """Evaluate the same already-seen test tasks for SAP diagnostics only."""
        task_accuracies = []
        training_states = {module: module.training for module in self.net.modules()}
        try:
            self.net.eval()
            with torch.no_grad():
                for task_id, test_loader in enumerate(dataset.test_loaders):
                    correct = 0
                    sample_count = 0
                    for data in test_loader:
                        inputs = data[0].to(self.device)
                        labels = data[1].to(self.device)
                        predictions = self.net(inputs)[:, :self.n_seen_classes].argmax(dim=1)
                        correct += (predictions == labels).sum().item()
                        sample_count += labels.numel()
                    if sample_count == 0:
                        raise ValueError(
                            f'seen test task {task_id} contains no diagnostic samples'
                        )
                    task_accuracies.append({
                        'task_id': task_id,
                        'sample_count': sample_count,
                        'accuracy': correct / sample_count,
                    })
        finally:
            for module, was_training in training_states.items():
                module.training = was_training
        return task_accuracies

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
        normalized_energy = energy / energy.sum()
        return projection, energy, normalized_energy, importance

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
