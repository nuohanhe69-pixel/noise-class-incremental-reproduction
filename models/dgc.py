from typing import Union
import copy
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from torch.nn import functional as F
from argparse import ArgumentParser
import numpy as np
from scipy import stats
from collections import deque

import math
from .er_ace_aer_abs import ErAceAerAbs
from utils.augmentations import apply_transform

def binary_search(left, right, func, target, tol, max_iter, is_func_decrease=True):
    for _ in range(max_iter):
        mid = (left + right) / 2
        if abs(func(mid) - target) < tol:
            return mid
        if (func(mid) > target) == is_func_decrease:
            left = mid   # 修正：增大 x 以减小递减函数的值
        else:
            right = mid  # 修正：减小 x 以增大递减函数的值
    return (left + right) / 2

class OptimizedGradientClippingBase(nn.Module):
    def __init__(self, 
                 epsilon=0.1, 
                 time_frame_s=100, 
                 queue_size_q=1000, 
                 current_queue_size=None, 
                 buffer_queue_size=None, 
                 min_queue_to_fit_current=None, 
                 min_queue_to_fit_buffer=None, 
                 binary_search_tol=1e-4, 
                 binary_search_max_iter=20, 
                 lazy_update_freq=10, 
                 use_single_queue=False): 
        super().__init__() 
        self.epsilon = epsilon 
        self.time_frame_s = time_frame_s 
        self.queue_size_q = queue_size_q 
        self.binary_search_tol = binary_search_tol 
        self.binary_search_max_iter = binary_search_max_iter 
        self.lazy_update_freq = lazy_update_freq 
        self.use_single_queue = use_single_queue 

        self.min_prob = 1e-7 
        self.max_prob = 1 - 1e-7 
        self.min_H = -math.log(self.max_prob) 
        self.max_H = -math.log(self.min_prob) 

        self.lr_t = 1.0 
        self.prob_thr_t = self.min_prob 
        self.tau_t = self.mapping_p_to_grad(torch.tensor(self.prob_thr_t)).abs().item() 

        # ---- fixed: explicit queue configs ---- 
        self.current_queue_size = current_queue_size if current_queue_size is not None else queue_size_q 
        self.buffer_queue_size = buffer_queue_size if buffer_queue_size is not None else queue_size_q 
        self.min_queue_to_fit_current = ( 
            min_queue_to_fit_current if min_queue_to_fit_current is not None 
            else max(32, self.current_queue_size // 4) 
        ) 
        self.min_queue_to_fit_buffer = ( 
            min_queue_to_fit_buffer if min_queue_to_fit_buffer is not None 
            else max(32, self.buffer_queue_size // 4) 
        ) 

        self.current_queue_H = deque(maxlen=self.current_queue_size) 
        self.buffer_queue_H = deque(maxlen=self.buffer_queue_size) 

        self.tau_current = None 
        self.tau_buffer = None 
        self.prob_thr_current = self.min_prob 
        self.prob_thr_buffer = self.min_prob 
        self.update_count_current = 0 
        self.update_count_buffer = 0 

 

        self.sample_H = np.linspace(self.min_H, self.max_H, 1000) 
        self.sample_p = self.mapping_H_to_p(torch.from_numpy(self.sample_H)).detach().numpy() 
        self.sample_grad = self.mapping_p_to_grad(torch.from_numpy(self.sample_p)).detach().numpy() 
        self.sample_grad_abs = np.abs(self.sample_grad) 

        # ---- fixed: separate GMMs for current / buffer ---- 
        self.gmm_current = self._build_gmm() 
        self.gmm_buffer = self._build_gmm() 

    def _build_gmm(self): 
        return GaussianMixture( 
            n_components=2, 
            max_iter=200, 
            tol=1e-3, 
            reg_covar=1e-4, 
            n_init=20, 
            warm_start=True, 
            init_params='kmeans' 
        ) 

    def _fit_gmm_internal(self, data, is_buffer=False): 
        gmm = self.gmm_buffer if is_buffer else self.gmm_current 
        try: 
            gmm.fit(data) 
        except Exception: 
            gmm = self._build_gmm() 
            gmm.fit(data) 

        if is_buffer: 
            self.gmm_buffer = gmm 
        else: 
            self.gmm_current = gmm 

        mean_index = np.argsort(gmm.means_, axis=0) 
        mean_clean = gmm.means_[mean_index[0]][0] 
        covar_clean = gmm.covariances_[mean_index[0]][0][0] 
        mean_noisy = gmm.means_[mean_index[1]][0] 
        covar_noisy = gmm.covariances_[mean_index[1]][0][0] 

        return mean_clean, covar_clean, mean_noisy, covar_noisy, gmm.weights_ 

    def mapping_p_to_H(self, p_y_hat): 
        return -torch.log(p_y_hat) 

    def mapping_H_to_p(self, H): 
        return torch.exp(-H) 

    def add_H_to_queue(self, prob_y_hat, is_buffer=False): 
        if self.use_single_queue: 
            is_buffer = False 

        H_gpu = self.mapping_p_to_H(prob_y_hat) 
        H_cpu = H_gpu.detach().cpu().numpy().reshape(-1) 
        if is_buffer: 
            self.buffer_queue_H.extend(H_cpu.tolist()) 
        else: 
            self.current_queue_H.extend(H_cpu.tolist()) 

    def get_tau_t(self, current_lr, is_buffer=False, epsilon_mult=1.0):
        if self.use_single_queue:
            is_buffer = False

        self.lr_t = current_lr
        queue = self.buffer_queue_H if is_buffer else self.current_queue_H

        if is_buffer:
            self.update_count_buffer += 1
            if self.update_count_buffer % self.lazy_update_freq != 0 and self.tau_buffer is not None:
                return self.prob_thr_buffer, self.tau_buffer
        else:
            self.update_count_current += 1
            if self.update_count_current % self.lazy_update_freq != 0 and self.tau_current is not None:
                return self.prob_thr_current, self.tau_current

        min_fit = self.min_queue_to_fit_buffer if is_buffer else self.min_queue_to_fit_current
        if len(queue) < min_fit:
            if is_buffer:
                return self.prob_thr_buffer, self.tau_buffer if self.tau_buffer is not None else self.tau_t
            else:
                return self.prob_thr_current, self.tau_current if self.tau_current is not None else self.tau_t

        queue_H = np.array(list(queue)).reshape(-1, 1)
        mean_clean, covar_clean, mean_noisy, covar_noisy, gmm_weights = self._fit_gmm_internal(
            queue_H, is_buffer=is_buffer
        ) 

        loc_clean, scale_clean = mean_clean, np.sqrt(covar_clean) 
        loc_noisy, scale_noisy = mean_noisy, np.sqrt(covar_noisy) 

        p_H_clean_vals = stats.norm.pdf(self.sample_H, loc_clean, scale_clean if scale_clean > 1e-6 else 1e-6) 
        p_H_noisy_vals = stats.norm.pdf(self.sample_H, loc_noisy, scale_noisy if scale_noisy > 1e-6 else 1e-6) 

        self.p_H_clean = p_H_clean_vals / (p_H_clean_vals.sum() + 1e-9) 
        self.p_H_noisy = p_H_noisy_vals / (p_H_noisy_vals.sum() + 1e-9) 

        target_val = 1 + self.lr_t * self.epsilon * epsilon_mult 

        prob_thr_t = binary_search( 
            self.min_prob, 
            self.max_prob, 
            self.calc_grad_ratio, 
            target_val, 
            self.binary_search_tol, 
            self.binary_search_max_iter, 
            is_func_decrease=True 
        ) 
        tau_t = self.mapping_p_to_grad(torch.tensor(prob_thr_t)).abs().item() 

        if is_buffer: 
            self.prob_thr_buffer = prob_thr_t 
            self.tau_buffer = tau_t 
        else: 
            self.prob_thr_current = prob_thr_t 
            self.tau_current = tau_t 

        return prob_thr_t, tau_t 

    def calc_grad_ratio(self, prob): 
        grad = self.mapping_p_to_grad(torch.tensor(prob)).item() 
        tau = np.abs(grad) 

        e_grad_noisy_mean = np.mean(self.p_H_noisy * np.minimum(self.sample_grad_abs, tau)) 
        e_grad_clean_mean = np.mean(self.p_H_clean * np.minimum(self.sample_grad_abs, tau)) 

        if e_grad_clean_mean < 1e-9: 
            return float('inf') 

        return e_grad_noisy_mean / e_grad_clean_mean


    def forward(self, logits, labels, lr, tau=None):
        # This forward method will calculate the OGC-modified loss
        # First, update tau_t based on current LR and queue_H
        if tau is not None:
            self.tau_t = tau
        else:
            self.prob_thr_t, self.tau_t = self.get_tau_t(lr)

        # Calculate the base loss (e.g., CE or FL)
        prob = F.softmax(logits, dim=1).clamp(min=self.min_prob, max=self.max_prob)
        p_y_hat = prob[torch.arange(logits.shape[0]), labels]

        # 1. 计算理论上的原始梯度并裁剪
        grad_p_y_hat = self.mapping_p_to_grad(p_y_hat)
        clipped_grad_p_y_hat = torch.clamp(grad_p_y_hat, -self.tau_t, self.tau_t)

        # 2. 计算正向传播的真实 Loss 数值 (用于打印和记录)
        loss_val = self.mapping_p_to_loss(p_y_hat).mean()
        
        # 3. 核心修复：构造伪损失 (Pseudo Loss)，将裁剪后的梯度强行注入计算图
        # 这一步保证了前向计算的 loss 不变，但反向传播时求导的结果严格等于 clipped_grad_p_y_hat
        pseudo_loss = (p_y_hat * clipped_grad_p_y_hat.detach()).mean()
        loss = loss_val.detach() + pseudo_loss - pseudo_loss.detach()

        return loss


class OGC_CE(OptimizedGradientClippingBase):
    def mapping_p_to_loss(self, p_y_hat):
        return -torch.log(p_y_hat)

    def mapping_p_to_grad(self, p_y_hat):
        return -1 / p_y_hat

class OGC_FL(OptimizedGradientClippingBase):
    def __init__(self, gamma=2.0, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)
        

    def mapping_p_to_loss(self, p_y_hat):
        return -(1 - p_y_hat)**self.gamma * p_y_hat.log()

    def mapping_p_to_grad(self, p_y_hat):
        # Derivative of FL w.r.t. p_y_hat
        # d/dp [-(1-p)^gamma * log(p)]
        # = - [ gamma * (1-p)^(gamma-1) * (-1) * log(p) + (1-p)^gamma * (1/p) ]
        # = gamma * (1-p)^(gamma-1) * log(p) - (1-p)^gamma / p
        return self.gamma * (1 - p_y_hat)**(self.gamma - 1) * p_y_hat.log() - (1 - p_y_hat)**self.gamma / p_y_hat

# OGC GCE Loss (Generalized Cross Entropy) - added for completeness if you want to use it
class OGC_GCE(OptimizedGradientClippingBase):
    def __init__(self, q=0.7, **kwargs):
        self.q = q
        super().__init__(**kwargs)
        

    def mapping_p_to_loss(self, p_y_hat):
        # L_GCE = (1 - p_y_hat^q) / q
        return (1 - p_y_hat**self.q) / self.q

    def mapping_p_to_grad(self, p_y_hat):
        # d/dp [(1 - p^q) / q] = (1/q) * (-q * p^(q-1)) = -p^(q-1)
        return -p_y_hat**(self.q - 1)


# --- 增强版CustomDataset ---
class CustomDatasetOGC(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None, extra=None, device="cpu"):
        self.device = device
        self.data = data.to(self.device)
        self.targets = targets.to(device) if targets is not None else None
        self.transform = transform
        self.probs = (torch.ones(len(self.data)) / len(self.data)).to(device)
        self.extra = extra
        self.ogc_weights = torch.ones(len(self.data)).to(device)  # OGC权重

    def set_probs(self, probs: Union[np.ndarray, torch.Tensor]):
        """
        Set the probability of each data point being correct (i.e., belonging to the Gaussian with the lowest mean)
        """
        if not isinstance(probs, torch.Tensor):
            probs = torch.tensor(probs)
        self.probs = probs.to(self.data.device)

    def set_ogc_weights(self, weights: Union[np.ndarray, torch.Tensor]):
        """
        Set the OGC weights for each data point
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)
        self.ogc_weights = weights.to(self.data.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the data, the target, the extra information (if any), the not augmented data,
        the probability of the data point being correct, and the OGC weight.

        Returns:
        - data: the augmented data
        - target: the target
        - extra: (optional) additional information
        - not_aug_data: the data without augmentation
        - prob: the probability of the data point being correct (from GMM)
        - ogc_weight: the OGC weight for this data point
        """
        not_aug_data = self.data[idx]
        data = not_aug_data.clone()
        if self.transform:
            data = apply_transform(data, self.transform, autosqueeze=True)
        ret = (data, self.targets[idx],)
        if self.extra is not None:
            ret += (self.extra[idx],)
        ret += (not_aug_data,)
        return ret + (self.probs[idx], self.ogc_weights[idx])


# --- Baseline + DGC model (inherits from ErAceAerAbs) ---
class DGC(ErAceAerAbs):
    NAME = 'dgc'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Call parent's get_parser to add all original ErAceAerAbs arguments
        parser = super(DGC, DGC).get_parser(parser)

        # Add OGC specific arguments
        ogc_group = parser.add_argument_group('Optimized Gradient Clipping (OGC)')
        ogc_group.add_argument('--ogc_loss_type', default='ce', type=str,
                            choices=['ce', 'fl', 'gce'], help='OGC loss type')
        ogc_group.add_argument('--ogc_epsilon', default=20, type=float,
                            help='OGC epsilon parameter')
        ogc_group.add_argument('--ogc_time_frame', default=32, type=int,
                            help='OGC time frame parameter')
        ogc_group.add_argument('--ogc_queue_size', default=2048,type=int,
                            help='OGC queue size parameter')
        ogc_group.add_argument('--ogc_gamma', default=2.0, type=float,
                            help='OGC focal loss gamma parameter (if ogc_loss_type is fl)')
        ogc_group.add_argument('--ogc_q', default=0.7, type=float,
                            help='OGC GCE q parameter (if ogc_loss_type is gce)')
        ogc_group.add_argument('--ogc_loss_weight', default=0.3, type=float,
                            help='Weight for the OGC loss component in total loss (1-weight for CE)')
        ogc_group.add_argument('--ogc_warmup_epochs', default=10, type=int,
                            help='Number of epochs for OGC parameter warm-up per task')
        ogc_group.add_argument('--ogc_lazy_update', default=10, type=int,
                            help='Frequency of GMM fitting and tau update (lazy update)')
        ogc_group.add_argument('--ogc_low_conf_weight', default=0.3, type=float,
                            help='Weight assigned to low confidence (noisy) samples by OGC')
        ogc_group.add_argument('--ogc_buffer_penalty_coeff', default=2, type=float,
                            help='Penalty coefficient for buffer sample selection (higher = stricter filtering)')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        # Initialize parent class (ErAceAerAbs) first
        super(DGC, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Initialize OGC loss function
        ogc_kwargs = {
            'epsilon': args.ogc_epsilon,
            'time_frame_s': args.ogc_time_frame,
            'queue_size_q': args.ogc_queue_size,
            'binary_search_tol': 1e-4,
            'binary_search_max_iter': 50,
            'lazy_update_freq': args.ogc_lazy_update
        }

        if args.ogc_loss_type == 'ce':
            self.ogc_loss_fn = OGC_CE(**ogc_kwargs)
        elif args.ogc_loss_type == 'fl':
            ogc_kwargs['gamma'] = args.ogc_gamma
            self.ogc_loss_fn = OGC_FL(**ogc_kwargs)
        elif args.ogc_loss_type == 'gce':
            ogc_kwargs['q'] = args.ogc_q
            self.ogc_loss_fn = OGC_GCE(**ogc_kwargs)
        else:
            raise ValueError(f"Unknown OGC loss type: {args.ogc_loss_type}")

        self.ogc_loss_weight = args.ogc_loss_weight
        self.current_batch_idx = 0

    @torch.no_grad()
    def load_buffer(self, buffer):
        """Restore task-boundary state omitted by legacy safe checkpoints."""
        super().load_buffer(buffer)

        selector = getattr(self.buffer, 'sample_selection_fn', None)
        selection_state_restored = bool(
            getattr(self.buffer, '_checkpoint_selection_state_restored', False)
        )
        if (
            selector is not None
            and hasattr(selector, 'importance_scores')
            and not self.buffer.is_empty()
            and not selection_state_restored
        ):
            all_buffer_data = self.buffer.get_all_data(device=self.device)
            images, labels = all_buffer_data[0], all_buffer_data[1]
            rebuilt_scores = []
            was_training = self.net.training
            self.net.eval()
            for start in range(0, len(images), self.args.batch_size):
                normalized = apply_transform(
                    images[start:start + self.args.batch_size],
                    self.normalization_transform,
                )
                logits = self.net(normalized)
                rebuilt_scores.append(
                    self.loss(
                        logits,
                        labels[start:start + self.args.batch_size],
                        reduction='none',
                    ).detach()
                )
            self.net.train(was_training)

            rebuilt_scores = torch.cat(rebuilt_scores)
            selector.importance_scores.fill_(rebuilt_scores.mean())
            selector.importance_scores[:len(rebuilt_scores)] = rebuilt_scores.to(
                selector.importance_scores.device,
            )
            if not torch.isfinite(selector.importance_scores).all():
                raise ValueError('failed to rebuild finite ABS importance scores from checkpoint')

        # AER reloads this snapshot at fitting epochs.  It is not part of old
        # checkpoints, so reconstruct it from the just-loaded task-boundary net.
        self.past_model_ckpt = copy.deepcopy(self.net.state_dict())
        if self.n_seen_classes > 0:
            self.seen_so_far = torch.arange(
                self.n_seen_classes,
                dtype=torch.long,
                device=self.device,
            )

    @torch.no_grad() 
    def _reset_ogc_state_for_new_task(self): 
        self.ogc_loss_fn.current_queue_H.clear() 
        self.ogc_loss_fn.tau_current = None 
        self.ogc_loss_fn.prob_thr_current = self.ogc_loss_fn.min_prob 
        self.ogc_loss_fn.update_count_current = 0 
    
        self.ogc_loss_fn.buffer_queue_H.clear() 
        self.ogc_loss_fn.tau_buffer = None 
        self.ogc_loss_fn.prob_thr_buffer = self.ogc_loss_fn.min_prob 
        self.ogc_loss_fn.update_count_buffer = 0 
    
        if self.buffer.is_empty(): 
            return 
    
        was_training = self.net.training 
        self.net.eval() 
    
        all_buf_data = self.buffer.get_all_data(device=self.device) 
        all_buf_inputs, all_buf_labels = all_buf_data[0], all_buf_data[1] 
    
        bs = self.args.batch_size 
        for i in range(0, len(all_buf_inputs), bs): 
            x = all_buf_inputs[i:i + bs] 
            y = all_buf_labels[i:i + bs] 
    
            x_norm = apply_transform(x, self.normalization_transform) 
            logits_buf = self.net(x_norm) 
            prob_buf = F.softmax(logits_buf, dim=1).clamp( 
                min=self.ogc_loss_fn.min_prob, 
                max=self.ogc_loss_fn.max_prob 
            ) 
            p_y_hat_buf = prob_buf[torch.arange(logits_buf.size(0), device=self.device), y] 
            self.ogc_loss_fn.add_H_to_queue(p_y_hat_buf, is_buffer=True) 
    
        self.net.train(was_training) 
    
    def end_task(self, dataset): 
        super().end_task(dataset)
        self.last_epoch = -1
        


    def observe(self, inputs, labels, not_aug_inputs, epoch, true_labels=None, sample_ids=None, source_task_ids=None):
        if hasattr(self, 'eff_logger'):
            self.eff_logger.start_timer()
        # This method is largely rewritten to integrate OGC
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        # Task Start Detection and Queue Reset
        if not hasattr(self, 'last_epoch'):
            self.last_epoch = -1
        
        if epoch == 0 and (self.last_epoch != 0):
            self._reset_ogc_state_for_new_task()
        
        self.last_epoch = epoch

        current_lr = self.opt.param_groups[0]['lr']
        # self.ogc_loss_fn.get_tau_t(current_lr, is_buffer=True)

        # Warm-up Logic
        ogc_warmup_epochs = self.args.ogc_warmup_epochs
        if epoch < ogc_warmup_epochs:
            # Linear ramp up for weight: 0 -> target
            # Use max(0.01, ...) to avoid complete zero if desired, but 0 is fine.
            current_ogc_weight = self.ogc_loss_weight * (epoch / ogc_warmup_epochs)
            # Epsilon multiplier: 2.0 -> 1.0 (Loose -> Strict)
            epsilon_mult = 2.0 - (epoch / ogc_warmup_epochs)
        else:
            current_ogc_weight = self.ogc_loss_weight
            epsilon_mult = 1.0

        # Forward pass for current data
        logits = self.net(inputs)

        # Masking for current task classes (same as ErAceAerAbs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        # --- OGC Specific Calculations for Current Data ---
        with torch.no_grad():
            # Get logits for not-augmented inputs to calculate confidence for OGC
            not_aug_logits = self.net(apply_transform(not_aug_inputs, self.normalization_transform))
            not_aug_logits = not_aug_logits.masked_fill(mask == 0, torch.finfo(not_aug_logits.dtype).min)

            # Calculate probabilities for OGC queue and weights
            prob_for_ogc = F.softmax(not_aug_logits, dim=1).clamp(min=self.ogc_loss_fn.min_prob, max=self.ogc_loss_fn.max_prob)
            p_y_hat_current = prob_for_ogc[torch.arange(not_aug_logits.shape[0]), labels]

            # Add H values to OGC queue (Current)
            self.ogc_loss_fn.add_H_to_queue(p_y_hat_current, is_buffer=False)

            # Get current OGC threshold and tau_t (update lr_t in ogc_loss_fn)
            current_lr = self.opt.param_groups[0]['lr']
            
            if getattr(self.args, 'disable_dynamic_threshold', False):
                prob_thr_t = self.args.fixed_prob_threshold
                tau_t = self.ogc_loss_fn.mapping_p_to_grad(torch.tensor(prob_thr_t)).abs().item()
            else:
                prob_thr_t, tau_t = self.ogc_loss_fn.get_tau_t(current_lr, is_buffer=False, epsilon_mult=epsilon_mult)

            # Calculate OGC weights for current data
            ogc_weights_current = torch.ones_like(p_y_hat_current)
            low_conf_mask_current = p_y_hat_current <= prob_thr_t
            # Adaptive weight reduction based on noise level guess or simply less aggressive
            # 40% noise is high, so we need to be careful not to kill gradients of clean but hard samples.
            # 0.5 might be too high if noise is dominant, but too low if clean samples are hard.
            # Let's try 0.3 for high noise tolerance (suppress noise more)
            ogc_weights_current[low_conf_mask_current] = self.args.ogc_low_conf_weight

            if true_labels is not None:
                true_labels_device = true_labels.to(labels.device)
                noisy_mask_current = labels != true_labels_device
                grad_abs_current = self.ogc_loss_fn.mapping_p_to_grad(p_y_hat_current).abs()
                clipped_mask_current = grad_abs_current > tau_t

        # Calculate losses for current data
        ce_loss_current = self.loss(logits, labels, reduction='none') # Keep reduction='none' for potential weighting
        ogc_loss_current = self.ogc_loss_fn(logits, labels, current_lr, tau=tau_t) # Use calculated tau

        # combined_loss_current = (ogc_weights_current * ce_loss_current).mean() * (1 - current_ogc_weight) + \
        #                         ogc_loss_current * current_ogc_weight
        combined_loss_current =  (ogc_weights_current * ce_loss_current).mean() * (1 - current_ogc_weight) + \
                        ogc_loss_current * current_ogc_weight


        # --- Replay Loss (if buffer not empty) ---
        loss_re = torch.tensor(0., device=self.device)
        buf_indexes = None
        not_aug_buf_inputs = None
        buf_labels = None
        if not self.buffer.is_empty():
            buffer_batch = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, return_not_aug=True,
                not_aug_transform=self.normalization_transform)
            buf_indexes, not_aug_buf_inputs, buf_inputs, buf_labels = buffer_batch[:4]

            with torch.no_grad():
                not_aug_buf_logits = self.net(not_aug_buf_inputs)
                if self.args.sample_selection_strategy != 'reservoir':
                    loss_not_aug_re_ext = self.loss(not_aug_buf_logits, buf_labels, reduction='none')
                    self.buffer.sample_selection_fn.update(buf_indexes, loss_not_aug_re_ext)

            # Replay if AER is disabled or if epoch is odd (or last)
            if not self.args.use_aer or self.is_aer_fitting_epoch(epoch):
                buf_logits = self.net(buf_inputs)

                # --- OGC Specific Calculations for Buffer Data ---
                with torch.no_grad():
                    prob_for_ogc_buf = F.softmax(not_aug_buf_logits, dim=1).clamp(min=self.ogc_loss_fn.min_prob, max=self.ogc_loss_fn.max_prob)
                    p_y_hat_buf = prob_for_ogc_buf[torch.arange(not_aug_buf_logits.shape[0]), buf_labels]
                    # Add H values from buffer to OGC queue (Buffer Queue)
                    self.ogc_loss_fn.add_H_to_queue(p_y_hat_buf, is_buffer=True)

                    # Recalculate OGC threshold (Use buffer queue)
                    if getattr(self.args, 'disable_dynamic_threshold', False):
                        prob_thr_t = self.args.fixed_prob_threshold
                        tau_t = self.ogc_loss_fn.mapping_p_to_grad(torch.tensor(prob_thr_t)).abs().item()
                    else:
                        prob_thr_t, tau_t = self.ogc_loss_fn.get_tau_t(current_lr, is_buffer=True)

                    # Calculate OGC weights for buffer data
                    ogc_weights_buf = torch.ones_like(p_y_hat_buf)
                    low_conf_mask_buf = p_y_hat_buf <= prob_thr_t
                    ogc_weights_buf[low_conf_mask_buf] = self.args.ogc_low_conf_weight

                    if hasattr(self.buffer, 'true_labels'):
                        buf_true_labels = self.buffer.true_labels[buf_indexes].to(buf_labels.device)
                        noisy_mask_buf = buf_labels != buf_true_labels
                        grad_abs_buf = self.ogc_loss_fn.mapping_p_to_grad(p_y_hat_buf).abs()
                        clipped_mask_buf = grad_abs_buf > tau_t

                # Calculate losses for buffer data
                ce_loss_buf = self.loss(buf_logits, buf_labels, reduction='none')
                ogc_loss_buf = self.ogc_loss_fn(buf_logits, buf_labels, current_lr, tau=tau_t)

                # Combine CE and OGC loss for buffer data
                # Use current_ogc_weight (or maybe full weight for buffer? Let's use same)
                loss_re =  (ogc_weights_buf * ce_loss_buf).mean() * (1 - current_ogc_weight) + \
                          ogc_loss_buf * current_ogc_weight
                # loss_re =  ce_loss_buf.mean() * (1 - current_ogc_weight) + \
                #           ogc_loss_buf * current_ogc_weight

            if self.args.use_aer and epoch % 2 == 0:
                loss_re = torch.tensor(0., device=self.device) # No replay loss during even epochs for AER

        self._record_loss_trace(
            epoch=epoch, not_aug_inputs=not_aug_inputs, labels=labels, true_labels=true_labels,
            sample_ids=sample_ids, source_task_ids=source_task_ids,
            memory_inputs=not_aug_buf_inputs, memory_labels=buf_labels, memory_indexes=buf_indexes,
        )

        total_loss = combined_loss_current + loss_re
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        # --- Buffer Update (Sample Insertion) with OGC Weights ---
        with torch.no_grad():
            # Update buffer if not using AER or only during buffer forgetting epochs
            if not self.args.use_aer or (self.args.use_aer and not self.is_aer_fitting_epoch(epoch)):
                # The loss_not_aug_ext is already calculated above for current data
                # Re-calculate it to ensure it's fresh if needed, or use the one from the top
                # For buffer insertion, we use the CE loss on not-augmented data
                loss_not_aug_ext_for_buffer = self.loss(not_aug_logits, labels, reduction='none')

                # Combine CE loss with OGC weights for sample selection
                # Higher combined_scores mean less likely to be "clean" (low loss, high confidence)
                # Enhanced penalty coefficient: configured by args
                combined_scores_for_buffer_selection = loss_not_aug_ext_for_buffer * (self.args.ogc_buffer_penalty_coeff - ogc_weights_current)

                # Sample insertion: select samples with the lowest combined_scores
                _, clean_mask = torch.topk(combined_scores_for_buffer_selection,
                                         round((1 - self.args.alpha_sample_insertion) * inputs.shape[0]),
                                         largest=False)

                trace_enabled = self._should_store_buffer_metadata()
                buffer_source_task_ids = (
                    self._buffer_source_task_ids(source_task_ids, labels)
                    if trace_enabled else None
                )
                self.buffer.add_data(examples=not_aug_inputs[clean_mask],
                                     labels=labels[clean_mask],
                                     true_labels=true_labels[clean_mask] if true_labels is not None else None,
                                     task_labels=buffer_source_task_ids[clean_mask] if buffer_source_task_ids is not None else None,
                                     sample_ids=sample_ids[clean_mask] if trace_enabled and sample_ids is not None else None,
                                     sample_selection_scores=loss_not_aug_ext_for_buffer[clean_mask] if self.args.sample_selection_strategy != 'reservoir' else None)

        if hasattr(self, 'eff_logger'):
            self.eff_logger.end_timer_and_record(self._current_task, epoch, self.current_batch_idx)
        self.current_batch_idx += 1

        return total_loss.item()
