import copy
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.utils.data import DataLoader

from datasets.utils.continual_dataset import ContinualDataset
from models.er_ace_aer_abs import CustomDataset, ErAceAerAbs
from utils.augmentations import apply_transform
from utils.sap_core import activation_projection_based_unlearning
from utils.sap_model_utils import attach_sap_methods


class SapTestDataset(torch.utils.data.Dataset):
    """Dataset wrapper used by SAP; it returns only transformed data and labels."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = apply_transform(img, self.transform, autosqueeze=True)
        return img, target


class AerSap(ErAceAerAbs):
    """Er-ACE + AER/ABS with SAP only, for ablation against OGC and OGC+SAP."""

    NAME = 'aer_sap'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser = super(AerSap, AerSap).get_parser(parser)

        sap_group = parser.add_argument_group('SAP Integration in AER')
        sap_group.add_argument('--enable_sap', type=int, default=1, help='Enable SAP unlearning?')
        sap_group.add_argument('--sap_frequency_epochs', type=int, default=0,
                               help='Frequency of SAP execution in epochs (0 = end of task only)')
        sap_group.add_argument('--sap_retain_samples', type=int, default=400,
                               help='Number of clean samples to retain for SVD projection calculation')
        sap_group.add_argument('--sap_mode', nargs='+', default=['sap'], help='modes: baseline, gpm, sgp, sap')
        sap_group.add_argument('--sap_mode_forget', nargs='+', default=['sap'])
        sap_group.add_argument('--sap_projection_type', nargs='+', default=['Mr'],
                               choices=['baseline', 'Mr', 'I-Mf', 'Mr-Mi', 'I-(Mf-Mi)'],
                               help='Projection type')
        sap_group.add_argument('--sap_max_samples', type=int, default=2000, help='Max samples for SVD')
        sap_group.add_argument('--sap_start_layer', nargs='+', type=int, default=[2],
                               help='Start layer index for SAP projection')
        sap_group.add_argument('--sap_end_layer', nargs='+', type=int, default=[3],
                               help='End layer index for SAP projection')
        sap_group.add_argument('--sap_scale_coff', nargs='+', type=float, default=[1000.0],
                               help='Scale coefficient for SAP')
        sap_group.add_argument('--sap_projection_location', nargs='+', default=['post'],
                               choices=['pre', 'post', 'all'])
        sap_group.add_argument('--sap_project_classifier', type=int, default=1)

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(AerSap, self).__init__(backbone, loss, args, transform, dataset=dataset)
        if args.enable_sap:
            self.net = attach_sap_methods(self.net)

    def _get_sap_retain_samples(self):
        if 'cifar10' in self.dataset.NAME.lower():
            return 400
        if 'cifar100' in self.dataset.NAME.lower():
            return 1500
        return self.args.sap_retain_samples

    def _sync_aer_checkpoint_with_current_model(self):
        if getattr(self.args, 'use_aer', 0) and hasattr(self, 'past_model_ckpt'):
            self.past_model_ckpt = copy.deepcopy(self.net.state_dict())

    def apply_sap(self, dataset):
        if self.buffer.is_empty():
            return False

        all_buf_data = self.buffer.get_all_data(device="cpu")
        inputs, labels = all_buf_data[0], all_buf_data[1]

        not_aug_dataset = CustomDataset(inputs, labels, transform=self.normalization_transform)
        not_aug_loader = DataLoader(not_aug_dataset, batch_size=self.args.batch_size, shuffle=False)

        was_training_for_eval = self.net.training
        self.net.eval()
        _, amb_idxs, _ = self.split_data(test_loader=not_aug_loader, model=self.net)
        self.net.train(was_training_for_eval)

        all_idxs = np.arange(len(inputs))
        clean_idxs = np.setdiff1d(all_idxs, amb_idxs)

        if len(clean_idxs) == 0:
            print("No clean samples found in buffer for SAP. Skipping SAP.")
            return False
        if len(amb_idxs) == 0:
            print("No ambiguous samples found in buffer for SAP. Skipping SAP.")
            return False

        min_required_samples = max(10, self.num_classes)
        if len(clean_idxs) < min_required_samples:
            print(f"Too few clean samples ({len(clean_idxs)} < {min_required_samples}) for SAP. Skipping SAP.")
            return False

        sap_retain_samples = self._get_sap_retain_samples()
        if len(clean_idxs) > sap_retain_samples:
            np.random.shuffle(clean_idxs)
            clean_idxs = clean_idxs[:sap_retain_samples]

        print(f"AER-SAP Execution: {len(clean_idxs)} Clean (Retain Space), {len(amb_idxs)} Noisy (Forget Space).")

        corrupt_indices = torch.tensor(amb_idxs, device="cpu")
        clean_indices = torch.tensor(clean_idxs, device="cpu")

        inputs_corrupt = inputs[corrupt_indices]
        labels_corrupt = labels[corrupt_indices]
        inputs_clean = inputs[clean_indices]
        labels_clean = labels[clean_indices]

        clean_dataset = SapTestDataset(inputs_clean, labels_clean, transform=self.normalization_transform)
        corrupt_dataset = SapTestDataset(inputs_corrupt, labels_corrupt, transform=self.normalization_transform)
        full_dataset = SapTestDataset(inputs, labels, transform=self.normalization_transform)

        train_loaders = (
            DataLoader(full_dataset, batch_size=self.args.batch_size, shuffle=True),
            DataLoader(clean_dataset, batch_size=self.args.batch_size, shuffle=True),
            DataLoader(corrupt_dataset, batch_size=self.args.batch_size, shuffle=True),
        )
        val_loaders = (None, None, None)
        test_loader = dataset.test_loaders[-1] if dataset and hasattr(dataset, 'test_loaders') else None

        sap_args = Namespace(
            mode=self.args.sap_mode,
            mode_forget=self.args.sap_mode_forget,
            projection_type=self.args.sap_projection_type,
            start_layer=self.args.sap_start_layer,
            end_layer=self.args.sap_end_layer,
            scale_coff=self.args.sap_scale_coff,
            scale_coff_forget=getattr(self.args, 'sap_scale_coff_forget', self.args.sap_scale_coff),
            projection_location=self.args.sap_projection_location,
            retain_samples=len(inputs_clean),
            forget_samples=len(inputs_corrupt),
            max_batch_size=self.args.batch_size,
            max_samples=self.args.sap_max_samples,
            class_label_names=[],
            num_classes=self.num_classes,
            use_valset=False,
            gpm_eps=0.95,
            project_classifier=bool(self.args.sap_project_classifier),
        )

        was_training = self.net.training
        self.net.eval()
        try:
            unlearnt_model, _ = activation_projection_based_unlearning(
                sap_args, self.net, train_loaders, val_loaders, test_loader, full_dataset, self.device
            )
            self.net = unlearnt_model
            self.net.to(self.device)
            return True
        except Exception as exc:
            print(f"SAP Unlearning failed: {exc}")
            self.net.to(self.device)
            return False
        finally:
            self.net.train(was_training)

    @torch.no_grad()
    def split_data(self, test_loader: DataLoader, model: nn.Module):
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        model.eval()
        model = model.to(self.device)

        losses = torch.tensor([])
        for data in test_loader:
            inputs, targets = data[0], data[1]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = ce_loss(outputs, targets)
            losses = torch.cat([losses, loss.detach().cpu()])

        losses = (losses - losses.min()) / ((losses.max() - losses.min()) + torch.finfo(torch.float32).eps)
        input_loss = losses.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-3, reg_covar=1e-4, n_init=10)
        gmm.fit(input_loss)

        prob = gmm.predict_proba(input_loss)
        mean_index = np.argsort(gmm.means_, axis=0)
        prob = prob[:, mean_index]
        pred = prob.argmax(axis=1)

        correct_idx = np.where(pred == 0)[0]
        amb_idx = np.where(pred == 1)[0]

        return correct_idx, amb_idx, prob

    def end_epoch(self, epoch: int, dataset: ContinualDataset):
        super().end_epoch(epoch, dataset)

        if self.args.enable_sap and self.args.sap_frequency_epochs > 0:
            if (epoch + 1) % self.args.sap_frequency_epochs == 0:
                print(f"Executing Periodic AER-SAP at epoch {epoch + 1}...")
                applied = self.apply_sap(dataset)
                if applied:
                    self._sync_aer_checkpoint_with_current_model()

    def end_task(self, dataset):
        super().end_task(dataset)

        if self.args.enable_sap:
            print("Executing End-of-Task AER-SAP...")
            applied = self.apply_sap(dataset)
            if applied:
                self._sync_aer_checkpoint_with_current_model()
