"""Pure AER/ABS with the validated Power-Normalized Linear Oracle SAP."""

from __future__ import annotations

import copy
import csv
import io
import json
import logging
import math
import random
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.dgc_sap import (
    DgcSap,
    SAP_FAILED,
    SAP_ORACLE_EXECUTED,
    SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION,
    SAP_SKIPPED_INFERENCE,
)
from models.er_ace_aer_abs import ErAceAerAbs
from utils.checkpoints import save_mammoth_checkpoint
from utils.conf import base_path, get_checkpoint_path
from utils.sap import (
    collect_classifier_input_features,
    normalize_classifier_input_features,
    project_linear_weight,
    resolve_classifier_module,
)


SAP_SKIPPED_AFTER_SECOND_BOUNDARY = 'SAP_SKIPPED_AFTER_SECOND_BOUNDARY'
TASKWISE_SAP_STARTED = 'TASKWISE_SAP_STARTED'
TASK_REFERENCE_EMPTY = 'TASK_REFERENCE_EMPTY'


def arcface_training_logits(cosine: torch.Tensor, labels: torch.Tensor,
                            margin: float = 0.5) -> torch.Tensor:
    """Apply a target-only monotonic angular margin, then fixed scale 64."""
    if not 0.0 <= margin < math.pi:
        raise ValueError('arcface_margin must be in [0, pi)')
    active = cosine != torch.finfo(cosine.dtype).min
    if not torch.all(active.gather(1, labels.unsqueeze(1))):
        raise ValueError('ArcFace target class must be active')
    adjusted = cosine
    if margin != 0.0:
        target = cosine.gather(1, labels.unsqueeze(1)).squeeze(1)
        epsilon = torch.finfo(cosine.dtype).eps
        target = target.clamp(-1.0 + epsilon, 1.0 - epsilon)
        sine = (1.0 - target.square()).clamp_min(epsilon).sqrt()
        angular_target = target * math.cos(margin) - sine * math.sin(margin)
        # Beyond pi-m, the linear branch preserves the target's monotonic order.
        transformed = torch.where(
            target > math.cos(math.pi - margin), angular_target,
            target - math.sin(math.pi - margin) * margin,
        )
        adjusted = cosine.scatter(1, labels.unsqueeze(1), transformed.unsqueeze(1))
    scaled = adjusted.masked_fill(~active, 0) * 64
    return scaled.masked_fill(~active, torch.finfo(cosine.dtype).min)


def cosface_training_logits(cosine: torch.Tensor, labels: torch.Tensor,
                            margin: float = 0.35) -> torch.Tensor:
    """Subtract a target-only cosine margin, then apply fixed scale 64."""
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError('cosface_margin must be finite and nonnegative')
    active = cosine != torch.finfo(cosine.dtype).min
    if not torch.all(active.gather(1, labels.unsqueeze(1))):
        raise ValueError('CosFace target class must be active')
    adjusted = cosine
    if margin != 0.0:
        target = cosine.gather(1, labels.unsqueeze(1)) - margin
        adjusted = cosine.scatter(1, labels.unsqueeze(1), target)
    scaled = adjusted.masked_fill(~active, 0) * 64
    return scaled.masked_fill(~active, torch.finfo(cosine.dtype).min)


def normalized_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                             reduction: str = 'mean') -> torch.Tensor:
    active = logits != torch.finfo(logits.dtype).min
    if not torch.all(active.gather(1, labels.unsqueeze(1))):
        raise ValueError('NCE target class must be active')
    working_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    log_probs = F.log_softmax(working_logits.masked_fill(~active, -torch.inf), dim=1)
    negative_log_probs = -log_probs.masked_fill(~active, 0)
    numerator = negative_log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    denominator = negative_log_probs.sum(dim=1).clamp_min(torch.finfo(working_logits.dtype).eps)
    losses = numerator / denominator
    if reduction == 'none':
        return losses
    if reduction == 'mean':
        return losses.mean()
    if reduction == 'sum':
        return losses.sum()
    raise ValueError(f'unsupported NCE reduction: {reduction}')


class AerSap(ErAceAerAbs):
    """AER/ABS + Oracle Linear SAP, without DGC/OGC."""

    NAME = 'aer_sap'
    COMPATIBILITY = ['class-il', 'task-il']
    requires_sap_checkpoint_state = True

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = ErAceAerAbs.get_parser(parser)
        group = parser.add_argument_group('Power-Normalized Linear Oracle SAP')
        group.add_argument('--sap_batch_size', type=int, default=32)
        group.add_argument(
            '--cosine_inference', type=int, default=0, choices=[0, 1],
            help='Use bias-free cosine logits from seen classifier rows during evaluation.',
        )
        group.add_argument(
            '--training_loss', type=str, default='ce',
            choices=['ce', 'normalized_cosine_ce', 'scale_cosine_ce', 'nce', 'arcface_ce', 'cosface_ce'],
            help='Training loss for current and replay samples.',
        )
        group.add_argument(
            '--arcface_margin', type=float, default=0.5,
            help='Target angular margin for arcface_ce (training scale is fixed at 64).',
        )
        group.add_argument(
            '--cosface_margin', type=float, default=0.35,
            help='Target cosine margin for cosface_ce (training scale is fixed at 64).',
        )
        group.add_argument(
            '--nce_ace_scope', type=str, default='baseline', choices=['baseline', 'aligned'],
            help='Active-class scope for NCE training only.',
        )
        group.add_argument(
            '--scale_cosine_scoring_scale', type=int, default=1, choices=[1, 64],
            help='ABS scoring scale for scale_cosine_ce (training always uses scale 64).',
        )
        group.add_argument(
            '--sap_oracle_reference', type=int, default=1, choices=[1],
            help='Oracle-clean references at each task boundary.',
        )
        group.add_argument(
            '--sap_oracle_scale', type=float, default=300.0,
            help='SAP scale coefficient for the final Linear projection.',
        )
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        if args.sap_batch_size <= 0:
            raise ValueError('sap_batch_size must be positive')
        if args.sap_oracle_scale != 300.0:
            raise ValueError('current-task SAP requires sap_oracle_scale=300')
        self.sap_history = []
        self._pending_task1_sap = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self.args, 'cosine_inference', 0) or self.net.training:
            return self.net(x)

        classifier = resolve_classifier_module(self.net)
        captured = []

        def capture_input(_module, args):
            captured.append(args[0])

        handle = classifier.register_forward_pre_hook(capture_input)
        try:
            self.net(x)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].ndim != 2:
            raise ValueError('cosine inference requires one 2-D classifier input')
        features = captured[0]
        if features.shape[1] != classifier.in_features:
            raise ValueError('cosine inference feature width differs from classifier')
        n_seen_classes = int(self.n_seen_classes)
        if not 0 < n_seen_classes <= classifier.out_features:
            raise ValueError('cosine inference requires valid seen-class count')
        weight = classifier.weight[:n_seen_classes]
        scale = weight.norm(p=2, dim=1).mean()
        weight_norm = F.normalize(weight, p=2, dim=1)
        features_norm = F.normalize(features, p=2, dim=1)
        return scale * torch.matmul(features_norm, weight_norm.T)

    def _observe_training_logits(
        self, inputs: torch.Tensor, *, present=None, replay=False,
    ) -> torch.Tensor:
        training_loss = getattr(self.args, 'training_loss', 'ce')
        if training_loss in ('ce', 'nce'):
            logits = super()._observe_training_logits(inputs)
            if training_loss == 'nce' and self.args.nce_ace_scope == 'aligned':
                return self._mask_observe_logits(logits, present=present, replay=replay)
            return logits
        if training_loss == 'arcface_ce':
            return self._observe_cosine_logits(inputs, scale=1, present=present, replay=replay)
        if training_loss == 'cosface_ce':
            return self._observe_cosine_logits(inputs, scale=1, present=present, replay=replay)
        scale = 64 if training_loss == 'scale_cosine_ce' else 1
        return self._observe_cosine_logits(inputs, scale=scale, present=present, replay=replay)

    def _observe_cosine_logits(
        self, inputs: torch.Tensor, *, scale: int, present=None, replay=False,
    ) -> torch.Tensor:
        classifier = resolve_classifier_module(self.net)
        features = self.net(inputs, returnt='features')
        if features.ndim != 2 or features.shape[1] != classifier.in_features:
            raise ValueError('cosine training requires 2-D classifier-input features')
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(classifier.weight, p=2, dim=1)
        logits = torch.matmul(features_norm, weight_norm.T)
        if scale != 1:
            logits = logits * scale
        return self._mask_observe_logits(logits, present=present, replay=replay)

    def _mask_observe_logits(self, logits: torch.Tensor, *, present=None, replay=False) -> torch.Tensor:
        if present is not None:
            active = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            active[present] = True
            return logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        if replay:
            active = torch.arange(logits.shape[1], device=logits.device) < self.n_seen_classes
            return logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        return logits

    def _observe_scoring_logits(
        self, inputs: torch.Tensor, *, present=None, replay=False,
    ) -> torch.Tensor:
        training_loss = getattr(self.args, 'training_loss', 'ce')
        if training_loss in ('ce', 'nce'):
            return super()._observe_scoring_logits(inputs)
        if training_loss == 'arcface_ce':
            return self._observe_cosine_logits(inputs, scale=1, present=present, replay=replay)
        if training_loss == 'cosface_ce':
            return self._observe_cosine_logits(inputs, scale=1, present=present, replay=replay)
        if training_loss == 'scale_cosine_ce':
            return self._observe_cosine_logits(
                inputs, scale=self.args.scale_cosine_scoring_scale,
                present=present, replay=replay,
            )
        return self._observe_training_logits(inputs, present=present, replay=replay)

    def _compute_training_loss(self, logits, labels, reduction='mean'):
        if getattr(self.args, 'training_loss', 'ce') == 'arcface_ce':
            logits = arcface_training_logits(logits, labels, margin=self.args.arcface_margin)
            return super()._compute_training_loss(logits, labels, reduction=reduction)
        if getattr(self.args, 'training_loss', 'ce') == 'cosface_ce':
            logits = cosface_training_logits(logits, labels, margin=self.args.cosface_margin)
            return super()._compute_training_loss(logits, labels, reduction=reduction)
        if getattr(self.args, 'training_loss', 'ce') == 'nce':
            return normalized_cross_entropy(logits, labels, reduction=reduction)
        return super()._compute_training_loss(logits, labels, reduction=reduction)

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
    def _capture_seen_test_decisions(self, dataset) -> dict:
        """Capture all seen test tasks in loader order with stable identities."""
        seen_tasks = list(range(int(self.current_task) + 1))
        if len(dataset.test_loaders) < len(seen_tasks):
            raise ValueError('seen test loaders are unavailable')
        seen_end = int(dataset.get_offsets(self.current_task)[1])
        classifier = resolve_classifier_module(self.net)
        captured_features = []

        def capture_classifier_input(_module, args):
            features = args[0].detach()
            if features.ndim != 2 or features.shape[1] != classifier.in_features:
                raise ValueError('unexpected test classifier-input shape')
            captured_features.append(features)

        training_states = {module: module.training for module in self.net.modules()}
        handle = classifier.register_forward_pre_hook(capture_classifier_input)
        features, logits, labels, sample_ids, task_ids = [], [], [], [], []
        try:
            self.net.eval()
            for task_id in seen_tasks:
                loader = dataset.test_loaders[task_id]
                original_ids = torch.as_tensor(loader.dataset.indexes, dtype=torch.long)
                cursor = 0
                for batch_index, data in enumerate(loader):
                    if getattr(self.args, 'debug_mode', False) and batch_index > self.get_debug_iters():
                        break
                    before_count = len(captured_features)
                    batch_logits = self(data[0].to(self.device)).detach().cpu()
                    if len(captured_features) != before_count + 1:
                        raise ValueError('test classifier hook did not fire exactly once')
                    batch_labels = data[1].detach().cpu().long()
                    count = len(batch_labels)
                    if batch_logits.shape[1] < seen_end or len(captured_features[-1]) != count:
                        raise ValueError('test feature/logit batch is not aligned')
                    features.append(captured_features.pop().cpu())
                    logits.append(batch_logits)
                    labels.append(batch_labels)
                    sample_ids.append(original_ids[cursor:cursor + count])
                    task_ids.append(torch.full((count,), task_id, dtype=torch.long))
                    cursor += count
                if cursor == 0 or (not getattr(self.args, 'debug_mode', False) and cursor != len(original_ids)):
                    raise ValueError('test IDs and loader samples are not aligned')
        finally:
            handle.remove()
            for module, was_training in training_states.items():
                module.training = was_training

        raw_features = torch.cat(features)
        full_logits = torch.cat(logits)
        true_labels = torch.cat(labels)
        source_tasks = torch.cat(task_ids)
        class_predictions = full_logits[:, :seen_end].argmax(dim=1)
        task_predictions = torch.empty_like(class_predictions)
        per_task_class_il, per_task_task_il = [], []
        for task_id in seen_tasks:
            start_c, end_c = map(int, dataset.get_offsets(task_id))
            mask = source_tasks == task_id
            task_predictions[mask] = full_logits[mask, start_c:end_c].argmax(dim=1) + start_c
            per_task_class_il.append(float((class_predictions[mask] == true_labels[mask]).float().mean() * 100))
            per_task_task_il.append(float((task_predictions[mask] == true_labels[mask]).float().mean() * 100))
        return {
            'sample_ids': torch.cat(sample_ids), 'task_ids': source_tasks,
            'raw_features': raw_features, 'labels': true_labels,
            'logits': full_logits, 'predictions': class_predictions,
            'task_predictions': task_predictions,
            'seen_classes': list(range(seen_end)),
            'per_task_class_il': per_task_class_il,
            'per_task_task_il': per_task_task_il,
        }

    @staticmethod
    def _build_test_decision_stats(pre_sap: dict, post_sap: dict,
                                   accuracy: dict, *, classifier_has_bias: bool,
                                   tolerance: float = 1e-5) -> dict:
        for key in ('sample_ids', 'task_ids', 'labels', 'raw_features'):
            if not torch.equal(pre_sap[key], post_sap[key]):
                raise ValueError(f'pre/post test {key} are not aligned')
        if pre_sap['seen_classes'] != post_sap['seen_classes']:
            raise ValueError('pre/post seen classes differ')
        for candidate, decisions in (('pre_sap', pre_sap), ('post_sap', post_sap)):
            for metric in ('per_task_class_il', 'per_task_task_il'):
                expected = accuracy[candidate][metric]
                actual = decisions[metric]
                if len(actual) != len(expected) or any(abs(a - b) > tolerance for a, b in zip(actual, expected)):
                    raise ValueError(f'{candidate} saved decisions do not reproduce {metric}')
        return {
            'sample_count': len(pre_sap['labels']),
            'seen_classes': pre_sap['seen_classes'],
            'classifier_has_bias': bool(classifier_has_bias),
            'prediction_changed_count': int((pre_sap['predictions'] != post_sap['predictions']).sum()),
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
        return (results_root / dataset.SETTING / dataset.NAME / self.NAME
                / 'double_boundary_taskwise_sap_v1' / str(run_id)
                / f'boundary_task_{int(self.current_task)}')

    @staticmethod
    def _save_taskwise_artifacts(directory: Path, tensors: dict,
                                 reference_counts: dict, coverage: dict,
                                 accuracy: dict, decision_stats: dict) -> dict:
        files = {}
        for filename, value in tensors.items():
            if isinstance(value, torch.Tensor):
                stored = value.detach().cpu()
            elif isinstance(value, dict):
                stored = {key: tensor.detach().cpu() for key, tensor in value.items()}
            else:
                stored = value
            torch.save(stored, directory / filename)
            if isinstance(value, torch.Tensor):
                files[filename] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
            elif isinstance(value, dict):
                files[filename] = {
                    'type': 'network_state_dict',
                    'tensors': {
                        key: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
                        for key, tensor in value.items()
                    },
                }
            else:
                files[filename] = {'type': type(value).__name__}
        for filename, value in (
            ('reference_counts.json', reference_counts),
            ('coverage.json', coverage), ('accuracy.json', accuracy),
            ('test_decision_stats.json', decision_stats),
        ):
            (directory / filename).write_text(
                json.dumps(value, indent=2, sort_keys=True), encoding='utf-8',
            )
            files[filename] = {'type': 'json'}
        return files

    def _analyze_task1_four_settings(
        self, dataset, x_l2, trusted_task_ids, weight_before, weight_after,
        bias_before, pre_sap_decisions,
    ) -> tuple[dict, dict]:
        """Evaluate Task1 counterfactual weights without changing the SAP commit."""
        old_mask = trusted_task_ids.to(x_l2.device) == 0
        old_x = x_l2[old_mask]
        if not len(old_x):
            raise ValueError('Task1 four-setting analysis requires Task0 references')
        old_gram = old_x.T @ old_x
        old_matrix, _, _, _, old_eigenvectors, old_eigenvalues = (
            self._build_oracle_projection(old_gram, return_eigenvectors=True)
        )
        old_start, old_end = map(int, dataset.get_offsets(0))
        old_weight, _ = project_linear_weight(
            weight_before[old_start:old_end, :], old_matrix,
        )
        task0_only = weight_before.clone()
        task0_only[old_start:old_end, :] = old_weight
        both = weight_after.clone()
        both[old_start:old_end, :] = old_weight
        candidates = {
            'none': weight_before.clone(),
            'task0_only': task0_only,
            'task1_only': weight_after.clone(),
            'both': both,
        }
        classifier = resolve_classifier_module(self.net)
        tensors = {
            'counterfactual_G_task_0.pt': old_gram,
            'counterfactual_M_task_0.pt': old_matrix,
            'counterfactual_eigenvalues_task_0.pt': old_eigenvalues,
            'counterfactual_eigenvectors_task_0.pt': old_eigenvectors,
        }
        rows = []
        first_decisions = None
        missing_flag = object()
        original_flag = getattr(self.args, 'cosine_inference', missing_flag)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        try:
            self.args.cosine_inference = 1
            with torch.random.fork_rng():
                for name, weight in candidates.items():
                    self._install_classifier_candidate(classifier, weight, bias_before)
                    decisions = self._capture_seen_test_decisions(dataset)
                    if first_decisions is None:
                        first_decisions = decisions
                    for key in ('sample_ids', 'labels', 'task_ids', 'raw_features'):
                        if (not torch.equal(decisions[key], first_decisions[key])
                                or not torch.equal(decisions[key], pre_sap_decisions[key])):
                            raise ValueError(f'Task1 four-setting test {key} are not aligned')
                    class_il = decisions['per_task_class_il']
                    task_il = decisions['per_task_task_il']
                    if len(class_il) != 2 or len(task_il) != 2:
                        raise ValueError('Task1 four-setting analysis requires two test tasks')
                    rows.append({
                        'setting': name,
                        'Task0 Class-IL': class_il[0],
                        'Task1 Class-IL': class_il[1],
                        'Overall Class-IL': sum(class_il) / 2,
                        'Task0 Task-IL': task_il[0],
                        'Task1 Task-IL': task_il[1],
                        'Overall Task-IL': sum(task_il) / 2,
                    })
                    prefix = f'task1_four_settings_{name}'
                    tensors[f'{prefix}_W.pt'] = weight
                    for key, output_name in (
                        ('sample_ids', 'sample_ids'),
                        ('labels', 'true_labels'),
                        ('task_ids', 'source_task_ids'),
                        ('predictions', 'predictions'),
                        ('task_predictions', 'task_predictions'),
                        ('logits', 'logits'),
                    ):
                        tensors[f'{prefix}_{output_name}.pt'] = decisions[key]
        finally:
            self._install_classifier_candidate(classifier, weight_after, bias_before)
            if original_flag is missing_flag:
                delattr(self.args, 'cosine_inference')
            else:
                self.args.cosine_inference = original_flag
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        return {
            'boundary_task_id': 1,
            'inference': 'cosine',
            'projection_scope': 'analysis_only',
            'task0_projector_source': 'Task1 boundary historical Task0 reference',
            'settings': rows,
        }, tensors

    @staticmethod
    def _save_task1_four_settings_report(directory: Path, report: dict) -> dict:
        columns = (
            'setting', 'Task0 Class-IL', 'Task1 Class-IL', 'Overall Class-IL',
            'Task0 Task-IL', 'Task1 Task-IL', 'Overall Task-IL',
        )
        (directory / 'task1_four_settings.json').write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding='utf-8',
        )
        csv_text = io.StringIO()
        writer = csv.DictWriter(csv_text, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report['settings'])
        (directory / 'task1_four_settings.csv').write_text(
            csv_text.getvalue(), encoding='utf-8',
        )
        markdown = ['| ' + ' | '.join(columns) + ' |',
                    '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        for row in report['settings']:
            markdown.append('| ' + ' | '.join(
                row[column] if column == 'setting' else f'{row[column]:.2f}'
                for column in columns
            ) + ' |')
        (directory / 'task1_four_settings.md').write_text(
            '\n'.join(markdown) + '\n', encoding='utf-8',
        )
        return {filename: {'type': suffix} for filename, suffix in (
            ('task1_four_settings.json', 'json'),
            ('task1_four_settings.csv', 'csv'),
            ('task1_four_settings.md', 'markdown'),
        )}

    def _run_taskwise_sap(self, dataset) -> None:
        if not 0 <= int(self.current_task) < int(dataset.N_TASKS):
            raise ValueError('task-wise SAP requires a valid task boundary')
        if float(self.args.sap_oracle_scale) != 300.0:
            raise ValueError('current-task SAP requires sap_oracle_scale=300')
        self._record_sap_event(status=TASKWISE_SAP_STARTED, sap_alpha=300.0)
        history_start = len(self.sap_history)
        network_before = copy.deepcopy(self.net.state_dict())
        had_past = hasattr(self, 'past_model_ckpt')
        past_before = copy.deepcopy(self.past_model_ckpt) if had_past else None
        final_directory = self._taskwise_artifact_directory(dataset)
        stage_directory = None
        stage = 'reference_construction'
        try:
            (images, trusted_labels, trusted_task_ids, reference_stats,
             evidence) = self._build_oracle_reference_batches(
                dataset, return_task_ids=True, return_evidence=True,
            )
            count = int(len(images))
            if count == 0 or any(len(evidence[key]) != count for key in (
                'sample_ids', 'true_labels', 'observed_labels',
                'source_task_ids', 'is_old', 'reference_order',
            )):
                raise ValueError('reference evidence and images are not aligned')
            if not torch.equal(evidence['true_labels'], trusted_labels) or not torch.equal(
                evidence['source_task_ids'], trusted_task_ids,
            ) or not torch.equal(evidence['reference_order'], torch.arange(count)):
                raise ValueError('reference evidence differs from the actual reference')
            seen_tasks = list(range(int(self.current_task) + 1))
            coverage = self._build_reference_coverage(
                dataset, trusted_labels, trusted_task_ids, seen_tasks,
            )
            if int(self.current_task) in coverage['empty_tasks']:
                self._record_sap_event(status=TASK_REFERENCE_EMPTY,
                                       reference_task_id=int(self.current_task))
                raise ValueError(f'empty trusted current-task reference: {self.current_task}')

            stage = 'feature_collection'
            classifier = resolve_classifier_module(self.net)
            weight_before = classifier.weight.detach().clone()
            bias_before = classifier.bias.detach().clone() if classifier.bias is not None else None
            # CIFAR current images are uint8 [0,255], while Buffer examples
            # originate from ToTensor() and are float [0,1]. Concatenation in
            # the unchanged reference builder promotes the mixed batch to float.
            # Restore the current slice to [0,1] before normalizing both slices.
            feature_images = images
            if self.current_task > 0 and images.is_floating_point():
                feature_images = images.clone()
                feature_images[:reference_stats['reference_new_count']].div_(255)
            def image_batches():
                for batch_images, _ in self._normalized_batches(feature_images, trusted_labels):
                    yield batch_images
            x_raw = collect_classifier_input_features(
                self.net, image_batches(), total_images=count,
            )
            x_l2, feature_norm_stats = normalize_classifier_input_features(x_raw)
            if x_l2.shape != (count, weight_before.shape[1]):
                raise ValueError('classifier feature shape differs from weight')
            if weight_before.shape[0] != int(dataset.N_CLASSES):
                raise ValueError('classifier row count differs from dataset classes')

            stage = 'taskwise_projection'
            weight_after = weight_before.clone()
            projected = torch.zeros(len(weight_before), dtype=torch.bool,
                                    device=weight_before.device)
            tensors = {
                'network_pre_sap.pt': network_before,
                'reference_sample_ids.pt': evidence['sample_ids'],
                'reference_true_labels.pt': evidence['true_labels'],
                'reference_observed_labels.pt': evidence['observed_labels'],
                'reference_source_task_ids.pt': evidence['source_task_ids'],
                'reference_is_old.pt': evidence['is_old'],
                'reference_order.pt': evidence['reference_order'],
                'X_raw.pt': x_raw, 'X_l2.pt': x_l2,
                'W_before.pt': weight_before, 'bias_before.pt': bias_before,
            }
            task_id = int(self.current_task)
            mask = trusted_task_ids.to(x_l2.device) == task_id
            task_x_raw = x_raw[mask]
            task_x = x_l2[mask]
            gram = task_x.T @ task_x
            matrix, energy, _, importance, eigenvectors, eigenvalues = (
                self._build_oracle_projection(gram, return_eigenvectors=True)
            )
            start_c, end_c = map(int, dataset.get_offsets(task_id))
            task_weight, _ = project_linear_weight(
                weight_before[start_c:end_c, :], matrix,
            )
            weight_after[start_c:end_c, :] = task_weight
            projected[start_c:end_c] = True
            tensors.update({
                f'X_raw_task_{task_id}.pt': task_x_raw,
                f'X_l2_task_{task_id}.pt': task_x,
                f'G_task_{task_id}.pt': gram,
                f'eigenvalues_task_{task_id}.pt': eigenvalues,
                f'energy_task_{task_id}.pt': energy,
                f'eigenvectors_task_{task_id}.pt': eigenvectors,
                f'importance_task_{task_id}.pt': importance,
                f'M_task_{task_id}.pt': matrix,
            })
            if not torch.equal(weight_after[~projected], weight_before[~projected]):
                raise AssertionError('SAP changed non-current classifier rows')

            stage = 'candidate_evaluation'
            accuracy, decisions = {}, {}
            for name, weight in (('pre_sap', weight_before), ('post_sap', weight_after)):
                self._install_classifier_candidate(classifier, weight, bias_before)
                decisions[name] = self._capture_seen_test_decisions(dataset)
                accuracy[name] = self._summarize_evaluation(dataset, self)
            decision_stats = self._build_test_decision_stats(
                decisions['pre_sap'], decisions['post_sap'], accuracy,
                classifier_has_bias=bias_before is not None,
            )
            network_after = copy.deepcopy(self.net.state_dict())
            for key, value in network_before.items():
                if key not in ('net.classifier.weight', 'classifier.weight') and not torch.equal(
                    value, network_after[key],
                ):
                    raise AssertionError(f'SAP changed non-target network state: {key}')
            if bias_before is not None and not torch.equal(classifier.bias, bias_before):
                raise AssertionError('SAP changed classifier bias')
            tensors.update({
                'network_post_sap.pt': network_after,
                'W_after.pt': weight_after,
                'bias_after.pt': classifier.bias.detach().clone() if classifier.bias is not None else None,
            })
            for name, result in decisions.items():
                for key in ('sample_ids', 'task_ids', 'labels', 'raw_features',
                            'logits', 'predictions', 'task_predictions'):
                    tensors[f'test_{name}_{key}.pt'] = result[key]

            four_settings_report = None
            if task_id == 1:
                stage = 'task1_four_settings_analysis'
                four_settings_report, analysis_tensors = self._analyze_task1_four_settings(
                    dataset, x_l2, trusted_task_ids, weight_before, weight_after,
                    bias_before, decisions['pre_sap'],
                )
                tensors.update(analysis_tensors)
                for key, value in network_after.items():
                    if not torch.equal(self.net.state_dict()[key], value):
                        raise AssertionError(f'Task1 analysis changed formal network state: {key}')

            stage = 'artifact_save'
            final_directory.parent.mkdir(parents=True, exist_ok=True)
            if final_directory.exists():
                raise FileExistsError(f'boundary artifact already exists: {final_directory}')
            stage_directory = Path(tempfile.mkdtemp(
                prefix=f'.boundary_task_{self.current_task}_',
                dir=final_directory.parent,
            ))
            files = self._save_taskwise_artifacts(
                stage_directory, tensors, evidence['counts'], coverage,
                accuracy, decision_stats,
            )
            if four_settings_report is not None:
                files.update(self._save_task1_four_settings_report(
                    stage_directory, four_settings_report,
                ))
            stage = 'taskwise_commit'
            self._install_classifier_candidate(classifier, weight_after, bias_before)
            self.past_model_ckpt = copy.deepcopy(self.net.state_dict())
            if any(not torch.equal(self.past_model_ckpt[key], value)
                   for key, value in self.net.state_dict().items()):
                raise AssertionError('AER checkpoint does not match post-SAP network')
            self._record_sap_event(
                status=SAP_ORACLE_EXECUTED, sap_alpha=300.0,
                seen_tasks=seen_tasks, projection_scope='current_task_only',
                total_reference_count=count, reference_stats=reference_stats,
                feature_norm_stats=feature_norm_stats,
                accuracy=accuracy, coverage=coverage,
                artifact_output_directory=str(final_directory),
            )
            manifest = {
                'experiment_name': 'double_boundary_taskwise_sap_v1',
                'boundary_task_id': int(self.current_task),
                'seen_tasks': seen_tasks, 'sap_alpha': 300.0,
                'reference_count': count, 'projection_scope': 'current_task_only',
                'expected_sap': True, 'started': True, 'succeeded': True,
                'artifact_complete': True, 'valid': True,
                'pre_state': 'network_pre_sap.pt',
                'post_state': 'network_post_sap.pt',
                'files': files,
            }
            (stage_directory / 'manifest.json').write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8',
            )
            stage_directory.rename(final_directory)
            stage_directory = None
        except Exception as error:
            self.net.load_state_dict(network_before)
            if had_past:
                self.past_model_ckpt = past_before
            elif hasattr(self, 'past_model_ckpt'):
                delattr(self, 'past_model_ckpt')
            if stage_directory is not None:
                shutil.rmtree(stage_directory)
            del self.sap_history[history_start:]
            self._record_sap_event(
                status=SAP_FAILED, oracle_stage=stage,
                error_type=type(error).__name__, error_message=str(error),
            )
            raise

    def _run_task_boundary_sap(self, dataset) -> None:
        start_from = getattr(self.args, 'start_from', None)
        if (getattr(self.args, 'loadcheck', None) is not None
                and start_from is not None and self.current_task < start_from):
            self._record_sap_event(status=SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION)
            return
        if getattr(self.args, 'inference_only', False):
            self._record_sap_event(status=SAP_SKIPPED_INFERENCE)
            return
        self._run_taskwise_sap(dataset)

    def end_task(self, dataset):
        start_from = getattr(self.args, 'start_from', None)
        if (start_from is not None
                and self.current_task < start_from
                and getattr(self.args, 'loadcheck', None) is not None):
            self._record_sap_event(status=SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION)
            return
        super().end_task(dataset)
        if self.current_task == 1 and getattr(self.args, 'savecheck', None):
            history = getattr(self, '_task1_pre_sap_results', None)
            if history is None or len(history) != 3 or len(history[0]) != 1 or len(history[1]) != 1:
                raise ValueError('Task1 Pre-SAP checkpoint requires Task0 training history')
            self._pending_task1_sap = True
            try:
                checkpoint_name = str(
                    Path(get_checkpoint_path()) / f'{self.args.ckpt_name}_1_pre_sap'
                )
                save_mammoth_checkpoint(
                    1, int(dataset.N_TASKS), self.args, self,
                    results=history,
                    optimizer_st=self.opt.state_dict() if hasattr(self, 'opt') else None,
                    checkpoint_name=checkpoint_name,
                )
            finally:
                self._pending_task1_sap = False
        self._run_task_boundary_sap(dataset)

    def resume_pending_task1_sap(self, dataset) -> bool:
        """Finish the saved Task1 boundary before the Task2 training loop."""
        if not getattr(self, '_pending_task1_sap', False):
            return False
        if self.current_task != 2 or int(dataset.c_task) != 1:
            raise ValueError('Task1 Pre-SAP resume requires reconstructed Task1 loaders')
        self._current_task = 1
        try:
            self._run_taskwise_sap(dataset)
        finally:
            self._current_task = 2
        self._pending_task1_sap = False
        return True

    def serialize_sap_state(self) -> dict:
        """Preserve the completed boundary needed to begin the next task."""
        pending_task1_sap = getattr(self, '_pending_task1_sap', False)
        phase = 'task1_pre_sap' if pending_task1_sap else 'task_boundary'
        if int(self.current_task) == 1:
            if not any(event.get('task_id') == 0 and event.get('status') == SAP_ORACLE_EXECUTED
                       for event in self.sap_history):
                raise ValueError('cannot checkpoint an incomplete Task0 SAP boundary')
            if pending_task1_sap and any(
                event.get('task_id') == 1 and event.get('status') == SAP_ORACLE_EXECUTED
                for event in self.sap_history
            ):
                raise ValueError('Task1 Pre-SAP checkpoint already contains Task1 SAP')
            if not hasattr(self, 'past_model_ckpt') or self.past_model_ckpt is None:
                raise ValueError('completed SAP boundary is missing the AER checkpoint')
            if phase == 'task_boundary':
                net_state = self.net.state_dict()
                if (net_state.keys() != self.past_model_ckpt.keys()
                        or any(not torch.equal(value, self.past_model_ckpt[key])
                               for key, value in net_state.items())):
                    raise ValueError('AER checkpoint differs from the post-SAP network')
        return {
            'version': 1,
            'phase': phase,
            'next_task': 2 if pending_task1_sap else int(self.current_task),
            'n_seen_classes': int(self.n_seen_classes),
            'n_past_classes': int(self.n_past_classes),
            'past_model_ckpt': (
                {key: value.detach().cpu().clone()
                 for key, value in self.past_model_ckpt.items()}
                if getattr(self, 'past_model_ckpt', None) is not None else None
            ),
            'seen_so_far': self.seen_so_far.detach().cpu().clone(),
            'history': copy.deepcopy(self.sap_history),
        }

    def load_sap_state(self, state: dict) -> None:
        """Load only at the next task, after checkpoint task reconstruction."""
        if state.get('version') != 1:
            raise ValueError('unsupported aer-sap checkpoint state')
        phase = state.get('phase', 'task_boundary')
        if phase not in ('task_boundary', 'task1_pre_sap'):
            raise ValueError('unsupported aer-sap checkpoint phase')
        next_task = int(state['next_task'])
        if int(self.current_task) != next_task:
            raise ValueError(
                f'aer-sap checkpoint requires start_from={next_task}; '
                f'reconstructed next task is {self.current_task}'
            )
        saved_past = state.get('past_model_ckpt')
        if phase == 'task1_pre_sap':
            if next_task != 2 or saved_past is None:
                raise ValueError('invalid Task1 Pre-SAP checkpoint state')
            if saved_past.keys() != self.net.state_dict().keys():
                raise ValueError('Task1 Pre-SAP AER checkpoint keys differ from network')
            expected_past, expected_seen = map(int, self.get_offsets(1))
            if (int(state['n_past_classes']) != expected_past
                    or int(state['n_seen_classes']) != expected_seen):
                raise ValueError('Task1 Pre-SAP seen-class counters differ from task')
            successful_tasks = [event.get('task_id') for event in state['history']
                                if event.get('status') == SAP_ORACLE_EXECUTED]
            if successful_tasks.count(0) != 1 or 1 in successful_tasks:
                raise ValueError('Task1 Pre-SAP checkpoint has invalid SAP history')
        if next_task == 1:
            net_state = self.net.state_dict()
            if (saved_past is None or saved_past.keys() != net_state.keys()
                    or any(not torch.equal(saved_past[key], value.cpu())
                           for key, value in net_state.items())):
                raise ValueError('loaded AER checkpoint differs from post-SAP network')
        self.past_model_ckpt = (
            {key: saved_past[key].to(value.device).clone()
             for key, value in self.net.state_dict().items()}
            if saved_past is not None else None
        )
        self.seen_so_far = state['seen_so_far'].to(self.device).clone()
        self.sap_history = copy.deepcopy(state['history'])
        if phase == 'task1_pre_sap':
            self._n_seen_classes = int(state['n_seen_classes'])
            self._n_past_classes = int(state['n_past_classes'])
        self._pending_task1_sap = phase == 'task1_pre_sap'
