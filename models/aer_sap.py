"""Pure AER/ABS with the validated Power-Normalized Linear Oracle SAP."""

from __future__ import annotations

import copy
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from models.dgc_sap import (
    DgcSap,
    SAP_FAILED,
    SAP_ORACLE_EXECUTED,
    SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION,
    SAP_SKIPPED_INFERENCE,
)
from models.er_ace_aer_abs import ErAceAerAbs
from utils.conf import base_path
from utils.sap import (
    collect_classifier_input_features,
    normalize_classifier_input_features,
    project_linear_weight,
    resolve_classifier_module,
)


SAP_SKIPPED_FIRST_SESSION_ONLY = 'SAP_SKIPPED_FIRST_SESSION_ONLY'
TASKWISE_SAP_STARTED = 'TASKWISE_SAP_STARTED'
TASK_REFERENCE_EMPTY = 'TASK_REFERENCE_EMPTY'


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
                 'skip the SAP pipeline.',
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

    @staticmethod
    def _weight_stats(weight_before, weight_after) -> dict[str, float]:
        denominator = weight_before.norm().clamp_min(
            torch.finfo(weight_before.dtype).eps,
        )
        return {
            'relative_weight_delta': (
                (weight_after - weight_before).norm() / denominator
            ).item(),
            'weight_norm_ratio': (weight_after.norm() / denominator).item(),
        }

    @staticmethod
    def _install_classifier_candidate(classifier, weight, bias_before) -> None:
        with torch.no_grad():
            classifier.weight.copy_(
                weight.to(device=classifier.weight.device, dtype=classifier.weight.dtype),
            )
            if bias_before is not None:
                if classifier.bias is None:
                    raise ValueError('classifier bias disappeared during SAP evaluation')
                classifier.bias.copy_(
                    bias_before.to(
                        device=classifier.bias.device, dtype=classifier.bias.dtype,
                    ),
                )
        if bias_before is not None and not torch.equal(classifier.bias, bias_before):
            raise ValueError('classifier bias changed during SAP evaluation')

    @staticmethod
    def _summarize_evaluation(dataset, model) -> dict:
        per_task_class_il, per_task_task_il = dataset.evaluate(model, dataset)
        per_task_class_il = [float(value) for value in per_task_class_il]
        per_task_task_il = [float(value) for value in per_task_task_il]
        if not per_task_class_il or not per_task_task_il:
            raise ValueError('SAP evaluation returned no task accuracies')
        return {
            'class_il': sum(per_task_class_il) / len(per_task_class_il),
            'task_il': sum(per_task_task_il) / len(per_task_task_il),
            'per_task_class_il': per_task_class_il,
            'per_task_task_il': per_task_task_il,
        }

    @torch.no_grad()
    def _capture_task1_test_decisions(self, dataset) -> dict:
        """Capture raw classifier inputs and Task1 Class-IL decisions."""
        if int(self.current_task) != 0:
            raise ValueError('Task1 test diagnostics are only valid at task 0')
        if not dataset.test_loaders:
            raise ValueError('Task1 test loader is unavailable')
        start_c, seen_end = dataset.get_offsets()
        start_c, seen_end = int(start_c), int(seen_end)
        if start_c != 0 or seen_end <= 0:
            raise ValueError('Task1 Class-IL evaluation requires classes [0, seen_end)')

        classifier = resolve_classifier_module(self.net)
        captured_features = []

        def capture_classifier_input(_module, args):
            if not args:
                raise RuntimeError('classifier received no positional input')
            features = args[0].detach()
            if features.ndim != 2 or features.shape[1] != classifier.in_features:
                raise ValueError('unexpected Task1 test classifier-input shape')
            captured_features.append(features)

        training_states = {
            module: module.training for module in self.net.modules()
        }
        handle = classifier.register_forward_pre_hook(capture_classifier_input)
        logits_batches = []
        label_batches = []
        try:
            self.net.eval()
            for batch_index, data in enumerate(dataset.test_loaders[0]):
                if (
                    getattr(self.args, 'debug_mode', False)
                    and batch_index > self.get_debug_iters()
                ):
                    break
                inputs, labels = data[0], data[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long()
                logits = self(inputs)
                if logits.ndim != 2 or logits.shape[1] < seen_end:
                    raise ValueError('Task1 test logits do not cover all seen classes')
                logits_batches.append(logits.detach())
                label_batches.append(labels.detach())
        finally:
            handle.remove()
            for module, was_training in training_states.items():
                module.training = was_training

        if not captured_features or not logits_batches:
            raise ValueError('Task1 test diagnostic capture produced no samples')
        raw_features = torch.cat(captured_features, dim=0)
        logits = torch.cat(logits_batches, dim=0)
        labels = torch.cat(label_batches, dim=0)
        sample_count = int(labels.shape[0])
        if not (
            raw_features.shape[0] == sample_count
            and logits.shape[0] == sample_count
        ):
            raise ValueError('Task1 test features, labels, and logits are not aligned')
        predictions = logits[:, :seen_end].argmax(dim=1)
        correct = int((predictions == labels).sum().item())
        accuracy = correct / sample_count * 100.0
        return {
            'raw_features': raw_features.detach().cpu(),
            'labels': labels.detach().cpu(),
            'logits': logits.detach().cpu(),
            'predictions': predictions.detach().cpu(),
            'seen_classes': list(range(seen_end)),
            'accuracy': accuracy,
        }

    @staticmethod
    def _build_task1_test_decision_stats(
        pre_sap: dict,
        post_sap: dict,
        accuracy: dict,
        *,
        classifier_has_bias: bool,
        tolerance: float = 1e-6,
    ) -> dict:
        """Verify saved decisions reproduce Task1 evaluator accuracy."""
        for key in ('raw_features', 'labels', 'logits', 'predictions'):
            if len(pre_sap[key]) != len(post_sap[key]):
                raise ValueError(f'pre/post Task1 test {key} are not aligned')
        if not torch.equal(pre_sap['labels'], post_sap['labels']):
            raise ValueError('pre/post Task1 test labels differ')
        if not torch.equal(pre_sap['raw_features'], post_sap['raw_features']):
            raise ValueError('pre/post raw Task1 test features differ')
        if pre_sap['seen_classes'] != post_sap['seen_classes']:
            raise ValueError('pre/post seen-class decision semantics differ')

        pre_evaluator = float(accuracy['pre_sap']['per_task_class_il'][0])
        post_evaluator = float(accuracy['post_sap']['per_task_class_il'][0])
        pre_error = abs(float(pre_sap['accuracy']) - pre_evaluator)
        post_error = abs(float(post_sap['accuracy']) - post_evaluator)
        if pre_error > tolerance or post_error > tolerance:
            raise ValueError(
                'saved Task1 decisions do not reproduce candidate evaluation: '
                f'pre_error={pre_error:.6e}, post_error={post_error:.6e}'
            )
        return {
            'sample_count': int(len(pre_sap['labels'])),
            'seen_classes': pre_sap['seen_classes'],
            'classifier_has_bias': bool(classifier_has_bias),
            'pre_accuracy_recomputed': float(pre_sap['accuracy']),
            'post_accuracy_recomputed': float(post_sap['accuracy']),
            'pre_accuracy_evaluator': pre_evaluator,
            'post_accuracy_evaluator': post_evaluator,
            'pre_accuracy_absolute_error': pre_error,
            'post_accuracy_absolute_error': post_error,
            'prediction_changed_count': int((
                pre_sap['predictions'] != post_sap['predictions']
            ).sum().item()),
        }

    @staticmethod
    def _build_reference_coverage(
        dataset, trusted_labels, trusted_task_ids, seen_tasks,
    ) -> dict:
        labels_cpu = trusted_labels.detach().cpu().long()
        task_ids_cpu = trusted_task_ids.detach().cpu().long()
        seen_tasks = [int(task_id) for task_id in seen_tasks]
        unexpected_task_ids = sorted(
            set(task_ids_cpu.unique().tolist()).difference(seen_tasks)
        )
        if unexpected_task_ids:
            raise ValueError(
                f'trusted references contain unseen task ids: {unexpected_task_ids}'
            )
        seen_classes = []
        for task_id in seen_tasks:
            start_c, end_c = dataset.get_offsets(task_id)
            seen_classes.extend(range(int(start_c), int(end_c)))
        class_counts = {
            str(class_id): int((labels_cpu == class_id).sum().item())
            for class_id in seen_classes
        }
        task_counts = {
            str(task_id): int((task_ids_cpu == task_id).sum().item())
            for task_id in seen_tasks
        }
        tasks = {}
        empty_tasks = []
        for task_id in seen_tasks:
            start_c, end_c = dataset.get_offsets(task_id)
            expected_classes = list(range(int(start_c), int(end_c)))
            present_classes = sorted(
                int(value) for value in labels_cpu[task_ids_cpu == task_id].unique().tolist()
            )
            present_expected = set(expected_classes).intersection(present_classes)
            missing_classes = sorted(set(expected_classes).difference(present_classes))
            is_empty = task_counts[str(task_id)] == 0
            if is_empty:
                empty_tasks.append(task_id)
            tasks[str(task_id)] = {
                'expected_classes': expected_classes,
                'present_classes': present_classes,
                'missing_classes': missing_classes,
                'coverage': (
                    len(present_expected) / len(expected_classes)
                    if expected_classes else 0.0
                ),
                'empty': is_empty,
            }
        return {
            'class_counts': class_counts,
            'task_counts': task_counts,
            'tasks': tasks,
            'empty_tasks': empty_tasks,
        }

    def _taskwise_artifact_directory(self, dataset) -> Path:
        results_root = Path(getattr(self.args, 'results_path', 'results'))
        if not results_root.is_absolute():
            results_root = Path(base_path()) / results_root
        run_id = getattr(self.args, 'conf_jobnum', None)
        if not run_id:
            run_id = f"seed{int(getattr(self.args, 'seed', 0) or 0)}"
        return (
            results_root
            / dataset.SETTING
            / dataset.NAME
            / self.NAME
            / 'first_session_only_taskwise_sap_v1'
            / str(run_id)
        )

    def _save_taskwise_artifacts(
        self,
        dataset,
        *,
        weight_before,
        x_task1,
        trusted_labels,
        trusted_task_ids,
        reference_stats,
        coverage,
        task_grams,
        task_projections,
        weight_after,
        accuracy,
        manifest,
        test_decisions,
        classifier_bias_before,
        decision_stats,
    ) -> Path:
        output_directory = self._taskwise_artifact_directory(dataset)
        output_directory.mkdir(parents=True, exist_ok=True)
        tensors = {
            'W_before.pt': weight_before,
            'X_task1.pt': x_task1,
            'trusted_labels.pt': trusted_labels,
            'trusted_task_ids.pt': trusted_task_ids,
            'W_after.pt': weight_after,
            'task1_test_features_raw.pt': test_decisions['pre_sap']['raw_features'],
            'task1_test_labels.pt': test_decisions['pre_sap']['labels'],
            'task1_test_logits_pre_sap.pt': test_decisions['pre_sap']['logits'],
            'task1_test_logits_post_sap.pt': test_decisions['post_sap']['logits'],
            'task1_test_predictions_pre_sap.pt': (
                test_decisions['pre_sap']['predictions']
            ),
            'task1_test_predictions_post_sap.pt': (
                test_decisions['post_sap']['predictions']
            ),
        }
        if classifier_bias_before is not None:
            tensors['classifier_bias_before.pt'] = classifier_bias_before
        for filename, tensor in tensors.items():
            torch.save(tensor.detach().cpu(), output_directory / filename)
        for task_id, gram in sorted(task_grams.items()):
            torch.save(
                gram.detach().cpu(), output_directory / f'G_task_{task_id}.pt',
            )
        for task_id, projection in sorted(task_projections.items()):
            torch.save(
                projection.detach().cpu(), output_directory / f'M_task_{task_id}.pt',
            )
        (output_directory / 'coverage.json').write_text(
            json.dumps(coverage, indent=2, sort_keys=True), encoding='utf-8',
        )
        (output_directory / 'reference_stats.json').write_text(
            json.dumps(reference_stats, indent=2, sort_keys=True), encoding='utf-8',
        )
        (output_directory / 'accuracy.json').write_text(
            json.dumps(accuracy, indent=2, sort_keys=True), encoding='utf-8',
        )
        (output_directory / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8',
        )
        (output_directory / 'task1_test_decision_stats.json').write_text(
            json.dumps(decision_stats, indent=2, sort_keys=True), encoding='utf-8',
        )
        return output_directory

    def _run_first_session_taskwise_sap(self, dataset) -> None:
        """Apply Task-wise SAP once after Task 1's complete AER boundary."""
        self._record_sap_event(status=TASKWISE_SAP_STARTED)
        classifier = None
        weight_before = None
        bias_before = None
        stage = 'reference_construction'
        try:
            (
                all_images,
                trusted_labels,
                trusted_task_ids,
                reference_stats,
            ) = self._build_oracle_reference_batches(
                dataset, return_task_ids=True,
            )
            total_images = int(all_images.shape[0])
            if total_images == 0:
                raise ValueError('first-session task-wise SAP reference set is empty')
            if not (
                len(trusted_labels) == total_images
                and len(trusted_task_ids) == total_images
            ):
                raise ValueError('trusted images, labels, and task ids are not aligned')
            seen_tasks = list(range(int(self.current_task) + 1))

            stage = 'classifier_snapshot'
            classifier = resolve_classifier_module(self.net)
            weight_before = classifier.weight.detach().clone()
            bias_before = (
                classifier.bias.detach().clone()
                if classifier.bias is not None else None
            )

            stage = 'feature_collection'
            batches_with_labels = list(
                self._normalized_batches(all_images, trusted_labels),
            )

            def image_batches():
                for images, _labels in batches_with_labels:
                    yield images

            features = collect_classifier_input_features(
                self.net, image_batches(), total_images=total_images,
            )
            x_task1, feature_norm_stats = normalize_classifier_input_features(features)
            if x_task1.shape[0] != total_images:
                raise ValueError('normalized features are not aligned with the reference set')
            if x_task1.shape[1] != weight_before.shape[1]:
                raise ValueError(
                    'classifier input dimension does not match normalized features'
                )
            if weight_before.shape[0] != int(dataset.N_CLASSES):
                raise ValueError('classifier row count does not match dataset classes')

            stage = 'coverage'
            coverage = self._build_reference_coverage(
                dataset, trusted_labels, trusted_task_ids, seen_tasks,
            )
            logging.info(
                'Task-wise SAP reference: total=%d task_counts=%s missing_classes=%s '
                'empty_tasks=%s',
                total_images,
                coverage['task_counts'],
                {
                    task_id: task_stats['missing_classes']
                    for task_id, task_stats in coverage['tasks'].items()
                    if task_stats['missing_classes']
                },
                coverage['empty_tasks'],
            )
            if coverage['empty_tasks']:
                for task_id in coverage['empty_tasks']:
                    self._record_sap_event(
                        status=TASK_REFERENCE_EMPTY,
                        reference_task_id=int(task_id),
                    )
                raise ValueError(
                    f"empty trusted task references: {coverage['empty_tasks']}"
                )

            stage = 'taskwise_projection'
            weight_after_taskwise = weight_before.clone()
            task_grams = {}
            task_projections = {}
            task_projection_stats = {}
            task_ids_device = trusted_task_ids.to(x_task1.device)
            projected_row_mask = torch.zeros(
                weight_before.shape[0], dtype=torch.bool, device=weight_before.device,
            )
            for task_id in seen_tasks:
                task_features = x_task1[task_ids_device == task_id]
                task_gram = task_features.transpose(0, 1) @ task_features
                (
                    task_projection,
                    task_energy,
                    task_normalized_energy,
                    task_importance,
                ) = self._build_oracle_projection(task_gram)
                task_grams[task_id] = task_gram
                start_c, end_c = dataset.get_offsets(task_id)
                start_c, end_c = int(start_c), int(end_c)
                task_weight, task_weight_stats = project_linear_weight(
                    weight_before[start_c:end_c, :], task_projection,
                )
                weight_after_taskwise[start_c:end_c, :] = task_weight
                projected_row_mask[start_c:end_c] = True
                task_projections[task_id] = task_projection
                task_projection_stats[str(task_id)] = {
                    'reference_count': int(task_features.shape[0]),
                    'gram_trace': task_gram.trace().item(),
                    'normalized_energy_min': task_normalized_energy.min().item(),
                    'normalized_energy_median': task_normalized_energy.median().item(),
                    'normalized_energy_max': task_normalized_energy.max().item(),
                    'importance_min': task_importance.min().item(),
                    'importance_median': task_importance.median().item(),
                    'importance_max': task_importance.max().item(),
                    'importance_trace': task_importance.sum().item(),
                    'relative_weight_delta': task_weight_stats['relative_weight_delta'],
                    'weight_norm_ratio': task_weight_stats['weight_norm_ratio'],
                    'projection_shape': list(task_projection.shape),
                    'gram_eigenvalue_min': task_energy.min().item(),
                    'gram_eigenvalue_max': task_energy.max().item(),
                }
            if not torch.equal(
                weight_after_taskwise[~projected_row_mask],
                weight_before[~projected_row_mask],
            ):
                raise AssertionError('Task-wise SAP changed unseen classifier rows')
            taskwise_weight_stats = self._weight_stats(
                weight_before, weight_after_taskwise,
            )

            stage = 'candidate_evaluation'
            accuracy = {}
            test_decisions = {}
            for candidate_name, candidate_weight in (
                ('pre_sap', weight_before),
                ('post_sap', weight_after_taskwise),
            ):
                self._install_classifier_candidate(
                    classifier, candidate_weight, bias_before,
                )
                test_decisions[candidate_name] = (
                    self._capture_task1_test_decisions(dataset)
                )
                accuracy[candidate_name] = self._summarize_evaluation(dataset, self)
                logging.info(
                    'Task-wise SAP candidate %s results: %s',
                    candidate_name, accuracy[candidate_name],
                )

            decision_stats = self._build_task1_test_decision_stats(
                test_decisions['pre_sap'],
                test_decisions['post_sap'],
                accuracy,
                classifier_has_bias=bias_before is not None,
            )

            manifest = {
                'experiment_name': 'first_session_only_taskwise_sap_v1',
                'task_id': int(self.current_task),
                'seen_tasks': seen_tasks,
                'sap_alpha': float(self.args.sap_oracle_scale),
                'projection_target': 'classifier',
                'projection_scope': 'taskwise_seen_tasks',
                'global_candidate_built': False,
                'classifier_has_bias': bias_before is not None,
                'test_feature_type': 'raw_classifier_input',
                'test_feature_l2_normalized': False,
                'test_decision_semantics': 'class_il_seen_classes',
            }
            stage = 'artifact_save'
            output_directory = self._save_taskwise_artifacts(
                dataset,
                weight_before=weight_before,
                x_task1=x_task1,
                trusted_labels=trusted_labels,
                trusted_task_ids=trusted_task_ids,
                reference_stats=reference_stats,
                coverage=coverage,
                task_grams=task_grams,
                task_projections=task_projections,
                weight_after=weight_after_taskwise,
                accuracy=accuracy,
                manifest=manifest,
                test_decisions=test_decisions,
                classifier_bias_before=bias_before,
                decision_stats=decision_stats,
            )

            stage = 'taskwise_commit'
            self._install_classifier_candidate(
                classifier, weight_after_taskwise, bias_before,
            )
            self.past_model_ckpt = copy.deepcopy(self.net.state_dict())
            logging.info(
                'First-session Task-wise SAP selected candidate=taskwise artifacts=%s',
                output_directory,
            )

            task_accuracy_comparisons = []
            for task_id, (before, after) in enumerate(zip(
                accuracy['pre_sap']['per_task_class_il'],
                accuracy['post_sap']['per_task_class_il'],
            )):
                test_loader = dataset.test_loaders[task_id] \
                    if task_id < len(dataset.test_loaders) else None
                sample_count = None
                if test_loader is not None and hasattr(test_loader, 'dataset'):
                    sample_count = len(test_loader.dataset)
                task_accuracy_comparisons.append({
                    'task_id': task_id,
                    'sample_count': sample_count,
                    'accuracy_before': before / 100.0,
                    'accuracy_after': after / 100.0,
                    'accuracy_delta': (after - before) / 100.0,
                })
            max_bias_delta = (
                0.0 if bias_before is None
                else (classifier.bias.detach() - bias_before).abs().max().item()
            )
            primary_task_id = seen_tasks[-1]
            primary_gram = task_grams[primary_task_id]
            primary_stats = task_projection_stats[str(primary_task_id)]
            self._record_sap_event(
                status=SAP_ORACLE_EXECUTED,
                projection_location='pre',
                projection_target='classifier',
                projection_scope='taskwise_seen_tasks',
                seen_tasks=seen_tasks,
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
                gram_shape=list(primary_gram.shape),
                gram_trace=primary_gram.trace().item(),
                feature_norm_before_min=feature_norm_stats['before']['min'],
                feature_norm_before_median=feature_norm_stats['before']['median'],
                feature_norm_before_mean=feature_norm_stats['before']['mean'],
                feature_norm_before_max=feature_norm_stats['before']['max'],
                feature_norm_after_min=feature_norm_stats['after']['min'],
                feature_norm_after_median=feature_norm_stats['after']['median'],
                feature_norm_after_mean=feature_norm_stats['after']['mean'],
                feature_norm_after_max=feature_norm_stats['after']['max'],
                gram_eigenvalue_min=primary_stats['gram_eigenvalue_min'],
                gram_eigenvalue_max=primary_stats['gram_eigenvalue_max'],
                normalized_energy_min=primary_stats['normalized_energy_min'],
                normalized_energy_median=primary_stats['normalized_energy_median'],
                normalized_energy_max=primary_stats['normalized_energy_max'],
                sap_alpha=float(self.args.sap_oracle_scale),
                importance_min=primary_stats['importance_min'],
                importance_median=primary_stats['importance_median'],
                importance_max=primary_stats['importance_max'],
                importance_trace=primary_stats['importance_trace'],
                n_seen_classes=int(self.n_seen_classes),
                total_classifier_rows=int(classifier.weight.shape[0]),
                relative_weight_delta=taskwise_weight_stats['relative_weight_delta'],
                weight_norm_ratio=taskwise_weight_stats['weight_norm_ratio'],
                max_bias_delta=max_bias_delta,
                seen_task_accuracy_comparisons=task_accuracy_comparisons,
                seen_average_accuracy_before=accuracy['pre_sap']['class_il'] / 100.0,
                seen_average_accuracy_after=accuracy['post_sap']['class_il'] / 100.0,
                task_projection_stats=task_projection_stats,
                accuracy=accuracy,
                coverage=coverage,
                final_selected_candidate='taskwise',
                artifact_output_directory=str(output_directory),
            )
        except Exception as error:
            if classifier is not None and weight_before is not None:
                self._install_classifier_candidate(
                    classifier, weight_before, bias_before,
                )
            self._record_sap_event(
                status=SAP_FAILED,
                oracle_stage=stage,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            logging.exception(
                'First-session Task-wise Oracle SAP failed; pre-SAP classifier was restored.'
            )

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
        self._run_first_session_taskwise_sap(dataset)

    def end_task(self, dataset):
        super().end_task(dataset)
        if self.current_task != 0:
            self._record_sap_event(status=SAP_SKIPPED_FIRST_SESSION_ONLY)
            return
        self._run_task_boundary_sap(dataset)
