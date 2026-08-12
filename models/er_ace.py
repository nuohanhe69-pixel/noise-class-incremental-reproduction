# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable

from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import add_rehearsal_args, ArgumentParser
from utils.augmentations import apply_transform
from utils.buffer import Buffer
from utils.loss_trace import LossTraceRecorder, add_loss_trace_args


class CustomLinear(torch.nn.Module):
    def __init__(self, indim, outdim, weight=None):
        super(CustomLinear, self).__init__()
        self.L = torch.nn.Linear(indim, outdim, bias=False)
        if weight is not None:
            self.L.weight.data = Variable(weight)

        self.scale_factor = 10

    def forward(self, x: torch.Tensor):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        L_norm = torch.norm(self.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        cos_dist = torch.mm(x_normalized, self.L.weight.div(L_norm + 0.00001).transpose(0, 1))

        scores = self.scale_factor * (cos_dist)

        return scores


class ErACE(ContinualModel):
    """Continual learning via Experience Replay with asymmetric cross-entropy."""
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--task_free', type=binary_to_boolean_type, default=False, help='Enable task-free training (replay starts from second task)?.')
        parser.add_argument('--use_custom_classifier', type=binary_to_boolean_type, default=True, help='Use the custom classifier used in the original work.')
        add_loss_trace_args(parser)

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.use_custom_classifier:
            assert hasattr(backbone, 'classifier'), 'The backbone must have a classifier layer.'
            backbone.classifier = CustomLinear(backbone.classifier.in_features, backbone.classifier.out_features)
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.loss_trace_recorder = None
        if self.args.enable_loss_trace:
            self.loss_trace_recorder = LossTraceRecorder(
                self.args, dataset_name=self.dataset.NAME, model_name=self.NAME,
            )

    def end_task(self, dataset):
        if self.loss_trace_recorder is not None:
            self.loss_trace_recorder.flush()

    def _should_trace_loss(self, epoch: int) -> bool:
        if self.loss_trace_recorder is None:
            return False
        if self.args.loss_trace_task >= 0 and self.current_task != self.args.loss_trace_task:
            return False
        if epoch < self.args.loss_trace_start_epoch:
            return False
        return self.args.loss_trace_end_epoch < 0 or epoch <= self.args.loss_trace_end_epoch

    def _record_loss_trace(self, *, epoch, not_aug_inputs, labels, true_labels, sample_ids,
                           source_task_ids, memory_inputs=None, memory_labels=None, memory_indexes=None):
        if not self._should_trace_loss(epoch):
            return
        required_current = {
            'true_labels': true_labels, 'sample_ids': sample_ids, 'source_task_ids': source_task_ids,
        }
        missing = [name for name, value in required_current.items() if value is None]
        if missing:
            raise ValueError(f'Loss tracing requires current-batch metadata: {missing}')

        current_inputs = apply_transform(not_aug_inputs, self.normalization_transform)
        trace_inputs = current_inputs
        trace_labels = labels
        memory_count = 0 if memory_inputs is None else memory_inputs.shape[0]
        if memory_count:
            for attr_name in ('true_labels', 'task_labels', 'sample_ids'):
                if not hasattr(self.buffer, attr_name):
                    raise ValueError(f'Loss tracing requires buffer attribute `{attr_name}`')
            trace_inputs = torch.cat((current_inputs, memory_inputs), dim=0)
            trace_labels = torch.cat((labels, memory_labels), dim=0)

        was_training = self.net.training
        try:
            self.net.eval()
            with torch.no_grad():
                trace_logits = self.net(trace_inputs)[:, :self.n_seen_classes]
                trace_losses = self.loss(trace_logits, trace_labels, reduction='none')
        finally:
            self.net.train(was_training)

        current_count = labels.shape[0]
        memory_kwargs = {}
        if memory_count:
            metadata_indexes = memory_indexes.to(self.buffer.true_labels.device)
            memory_kwargs = {
                'memory_losses': trace_losses[current_count:],
                'memory_labels': memory_labels,
                'memory_true_labels': self.buffer.true_labels[metadata_indexes],
                'memory_sample_ids': self.buffer.sample_ids[metadata_indexes],
                'memory_source_task_ids': self.buffer.task_labels[metadata_indexes],
                'memory_buffer_slot_ids': memory_indexes,
            }
        self.loss_trace_recorder.record(
            task_id=self.current_task, epoch_id=epoch,
            epoch_iteration_id=self.epoch_iteration, task_iteration_id=self.task_iteration,
            num_batches_per_epoch=len(self.dataset.train_loader), replay_on=bool(memory_count),
            current_losses=trace_losses[:current_count], current_labels=labels,
            current_true_labels=true_labels, current_sample_ids=sample_ids,
            current_source_task_ids=source_task_ids, **memory_kwargs,
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, true_labels=None,
                sample_ids=None, source_task_ids=None):

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        # if self.seen_so_far.max() < (self.num_classes - 1):
        mask[:, self.seen_so_far.max():] = 1

        if self.current_task > 0 or self.args.task_free:
            logits = logits.masked_fill(mask == 0, -1e9)  # torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)
        buf_indexes = None
        not_aug_buf_inputs = None
        buf_labels = None

        if len(self.buffer) > 0:
            if self.args.task_free or self.current_task > 0:
                # sample from buffer
                buffer_batch = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device,
                    return_index=True, return_not_aug=True,
                    not_aug_transform=self.normalization_transform)
                buf_indexes, not_aug_buf_inputs, buf_inputs, buf_labels = buffer_batch[:4]
                loss_re = self.loss(self.net(buf_inputs), buf_labels)

                loss += loss_re

        self._record_loss_trace(
            epoch=epoch, not_aug_inputs=not_aug_inputs, labels=labels, true_labels=true_labels,
            sample_ids=sample_ids, source_task_ids=source_task_ids,
            memory_inputs=not_aug_buf_inputs, memory_labels=buf_labels, memory_indexes=buf_indexes,
        )

        loss.backward()
        self.opt.step()

        trace_enabled = self.loss_trace_recorder is not None
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             true_labels=true_labels if trace_enabled else None,
                             task_labels=source_task_ids if trace_enabled else None,
                             sample_ids=sample_ids if trace_enabled else None)

        return loss.item()
