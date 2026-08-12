"""Iteration-level per-sample loss tracing for noisy continual learning."""

import atexit
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


SAMPLE_FIELDS = (
    'run_id', 'task_id', 'epoch_id', 'epoch_iteration_id', 'task_iteration_id', 'x_epoch', 'replay_on',
    'source', 'sample_id', 'buffer_slot_id', 'source_task_id', 'observed_class_id', 'true_class_id',
    'is_noisy', 'loss', 'group',
)

CURVE_FIELDS = (
    'run_id', 'task_id', 'epoch_id', 'epoch_iteration_id', 'task_iteration_id', 'x_epoch', 'replay_on',
    'noisy_loss', 'hard_old_loss', 'easy_old_loss', 'new_loss',
    'noisy_count', 'hard_old_count', 'easy_old_count', 'new_count',
)


def add_loss_trace_args(parser):
    """Add the optional four-group loss tracing arguments to a model parser."""
    trace_group = parser.add_argument_group('Per-sample loss tracing')
    trace_group.add_argument('--enable_loss_trace', default=0, type=int, choices=[0, 1],
                             help='Record iteration-level per-sample CE losses and the four requested loss curves?')
    trace_group.add_argument('--loss_trace_task', default=1, type=int,
                             help='Zero-based task id to trace. Use -1 to trace every task.')
    trace_group.add_argument('--loss_trace_start_epoch', default=0, type=int,
                             help='First zero-based epoch to trace (inclusive).')
    trace_group.add_argument('--loss_trace_end_epoch', default=-1, type=int,
                             help='Last zero-based epoch to trace (inclusive). Use -1 for no upper bound.')
    trace_group.add_argument('--loss_trace_dir', default='output/loss_trace', type=str,
                             help='Directory in which to create the run-specific loss trace folder.')
    trace_group.add_argument('--loss_trace_flush_every', default=100, type=int,
                             help='Flush buffered loss trace rows every N recorded iterations.')
    trace_group.add_argument('--loss_trace_hard_ratio', default=0.2, type=float,
                             help='Fraction of clean old-memory samples assigned to hard_old per iteration.')
    return parser


def _safe_mean(values: torch.Tensor) -> float:
    return values.mean().item() if values.numel() else float('nan')


def assign_dynamic_groups(current_losses: torch.Tensor,
                          current_labels: torch.Tensor,
                          current_true_labels: torch.Tensor,
                          memory_losses: torch.Tensor,
                          memory_labels: torch.Tensor,
                          memory_true_labels: torch.Tensor,
                          memory_source_task_ids: torch.Tensor,
                          current_task_id: int,
                          hard_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    """Return mutually-exclusive masks for the four requested groups."""
    if not 0 < hard_ratio < 1:
        raise ValueError('loss_trace_hard_ratio must be strictly between 0 and 1')
    current_noisy = current_labels != current_true_labels
    old_memory = memory_source_task_ids < current_task_id
    memory_noisy = memory_labels != memory_true_labels
    old_noisy = old_memory & memory_noisy
    clean_old = old_memory & ~memory_noisy

    hard_old = torch.zeros_like(clean_old, dtype=torch.bool)
    clean_old_indexes = torch.where(clean_old)[0]
    if clean_old_indexes.numel():
        hard_count = max(1, math.ceil(hard_ratio * clean_old_indexes.numel()))
        hard_local_indexes = torch.topk(
            memory_losses[clean_old_indexes], k=hard_count, largest=True,
        ).indices
        hard_old[clean_old_indexes[hard_local_indexes]] = True

    return {
        'current_noisy': current_noisy,
        'current_new': ~current_noisy,
        'memory_noisy': old_noisy,
        'memory_hard_old': hard_old,
        'memory_easy_old': clean_old & ~hard_old,
    }


class LossTraceRecorder:
    """Buffered CSV writer for raw sample losses and per-iteration curves."""

    def __init__(self, args, dataset_name: str, model_name: str):
        self.run_id = str(getattr(args, 'conf_jobnum', 'manual-run'))
        self.flush_every = int(args.loss_trace_flush_every)
        if self.flush_every <= 0:
            raise ValueError('loss_trace_flush_every must be greater than zero')
        self.hard_ratio = float(getattr(args, 'loss_trace_hard_ratio', 0.2))
        if not 0 < self.hard_ratio < 1:
            raise ValueError('loss_trace_hard_ratio must be strictly between 0 and 1')

        safe_name = f'{dataset_name}_{model_name}_{self.run_id[:8]}'.replace('/', '-')
        self.output_dir = Path(args.loss_trace_dir).expanduser() / safe_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_path = self.output_dir / 'sample_losses.csv'
        self.curve_path = self.output_dir / 'iteration_curves.csv'
        self._sample_rows: List[dict] = []
        self._curve_rows: List[dict] = []
        self._recorded_iterations = 0

        training_parameters = {}
        for parameter_name in (
            'n_epochs', 'batch_size', 'minibatch_size', 'buffer_size', 'optimizer', 'lr',
            'optim_mom', 'optim_wd', 'optim_nesterov', 'use_aer',
            'sample_selection_strategy', 'alpha_sample_insertion', 'buffer_fitting_epochs',
        ):
            if hasattr(args, parameter_name):
                training_parameters[parameter_name] = getattr(args, parameter_name)

        metadata = {
            'run_id': self.run_id,
            'dataset': dataset_name,
            'model': model_name,
            'seed': getattr(args, 'seed', None),
            'noise_type': getattr(args, 'noise_type', None),
            'noise_rate': getattr(args, 'noise_rate', None),
            'hard_old_ratio': self.hard_ratio,
            'loss_definition': 'cross_entropy(logits_over_seen_classes, observed_label, reduction=none)',
            'loss_timing': 'before optimizer.step',
            'loss_input': 'normalized non-augmented input',
            'training_parameters': training_parameters,
            'grouping': {
                'noisy': 'current or old-memory sample with observed_label != true_label',
                'new': 'current sample with observed_label == true_label',
                'hard_old': f'highest-loss {self.hard_ratio:.1%} of clean old-memory samples in this sampled memory batch',
                'easy_old': f'remaining {1 - self.hard_ratio:.1%} of clean old-memory samples in this sampled memory batch',
            },
        }
        (self.output_dir / 'metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
        )
        atexit.register(self.flush)

    @staticmethod
    def _cpu(tensor: Optional[torch.Tensor], name: str) -> torch.Tensor:
        if tensor is None:
            raise ValueError(f'Loss tracing requires `{name}`')
        return tensor.detach().to('cpu')

    @staticmethod
    def _append_csv(path: Path, fieldnames: Iterable[str], rows: List[dict]) -> None:
        if not rows:
            return
        write_header = not path.exists()
        with path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def record(self, *, task_id: int, epoch_id: int, epoch_iteration_id: int, task_iteration_id: int,
               num_batches_per_epoch: int, replay_on: bool,
               current_losses: torch.Tensor, current_labels: torch.Tensor, current_true_labels: torch.Tensor,
               current_sample_ids: torch.Tensor, current_source_task_ids: torch.Tensor,
               memory_losses: Optional[torch.Tensor] = None, memory_labels: Optional[torch.Tensor] = None,
               memory_true_labels: Optional[torch.Tensor] = None, memory_sample_ids: Optional[torch.Tensor] = None,
               memory_source_task_ids: Optional[torch.Tensor] = None,
               memory_buffer_slot_ids: Optional[torch.Tensor] = None) -> None:
        current_losses = self._cpu(current_losses, 'current_losses')
        current_labels = self._cpu(current_labels, 'current_labels')
        current_true_labels = self._cpu(current_true_labels, 'current_true_labels')
        current_sample_ids = self._cpu(current_sample_ids, 'current_sample_ids')
        current_source_task_ids = self._cpu(current_source_task_ids, 'current_source_task_ids')

        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=current_losses.dtype)
        memory_losses = empty_float if memory_losses is None else self._cpu(memory_losses, 'memory_losses')
        memory_labels = empty_long if memory_labels is None else self._cpu(memory_labels, 'memory_labels')
        memory_true_labels = empty_long if memory_true_labels is None else self._cpu(memory_true_labels, 'memory_true_labels')
        memory_sample_ids = empty_long if memory_sample_ids is None else self._cpu(memory_sample_ids, 'memory_sample_ids')
        memory_source_task_ids = empty_long if memory_source_task_ids is None else self._cpu(memory_source_task_ids, 'memory_source_task_ids')
        memory_buffer_slot_ids = empty_long if memory_buffer_slot_ids is None else self._cpu(memory_buffer_slot_ids, 'memory_buffer_slot_ids')

        groups = assign_dynamic_groups(
            current_losses, current_labels, current_true_labels,
            memory_losses, memory_labels, memory_true_labels, memory_source_task_ids, task_id,
            hard_ratio=self.hard_ratio,
        )
        x_epoch = epoch_id + epoch_iteration_id / max(1, num_batches_per_epoch)
        common = {
            'run_id': self.run_id, 'task_id': task_id, 'epoch_id': epoch_id,
            'epoch_iteration_id': epoch_iteration_id, 'task_iteration_id': task_iteration_id,
            'x_epoch': x_epoch, 'replay_on': int(replay_on),
        }

        current_groups = torch.where(groups['current_noisy'], 0, 1)
        current_group_names = ('noisy', 'new')
        for index in range(current_losses.numel()):
            self._sample_rows.append({
                **common, 'source': 'current', 'sample_id': int(current_sample_ids[index]),
                'buffer_slot_id': -1, 'source_task_id': int(current_source_task_ids[index]),
                'observed_class_id': int(current_labels[index]), 'true_class_id': int(current_true_labels[index]),
                'is_noisy': int(groups['current_noisy'][index]), 'loss': float(current_losses[index]),
                'group': current_group_names[int(current_groups[index])],
            })

        memory_group_names = ['ignored_memory_current'] * memory_losses.numel()
        for group_name, mask_name in (
            ('noisy', 'memory_noisy'), ('hard_old', 'memory_hard_old'), ('easy_old', 'memory_easy_old'),
        ):
            for index in torch.where(groups[mask_name])[0].tolist():
                memory_group_names[index] = group_name
        for index in range(memory_losses.numel()):
            self._sample_rows.append({
                **common, 'source': 'memory', 'sample_id': int(memory_sample_ids[index]),
                'buffer_slot_id': int(memory_buffer_slot_ids[index]),
                'source_task_id': int(memory_source_task_ids[index]),
                'observed_class_id': int(memory_labels[index]), 'true_class_id': int(memory_true_labels[index]),
                'is_noisy': int(memory_labels[index] != memory_true_labels[index]),
                'loss': float(memory_losses[index]), 'group': memory_group_names[index],
            })

        noisy_losses = torch.cat((
            current_losses[groups['current_noisy']], memory_losses[groups['memory_noisy']],
        ))
        curve_values = {
            'noisy': noisy_losses,
            'hard_old': memory_losses[groups['memory_hard_old']],
            'easy_old': memory_losses[groups['memory_easy_old']],
            'new': current_losses[groups['current_new']],
        }
        self._curve_rows.append({
            **common,
            **{f'{name}_loss': _safe_mean(values) for name, values in curve_values.items()},
            **{f'{name}_count': values.numel() for name, values in curve_values.items()},
        })
        self._recorded_iterations += 1
        if self._recorded_iterations % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        self._append_csv(self.sample_path, SAMPLE_FIELDS, self._sample_rows)
        self._append_csv(self.curve_path, CURVE_FIELDS, self._curve_rows)
        self._sample_rows.clear()
        self._curve_rows.clear()
