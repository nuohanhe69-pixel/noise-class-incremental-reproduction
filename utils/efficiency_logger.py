import os
import time
import torch
import csv
from typing import Dict, Any, Optional

class EfficiencyLogger:
    def __init__(self, save_dir: str, model_name: str, run_id: str = 'default'):
        self.save_dir = save_dir
        self.model_name = model_name
        self.run_id = run_id
        self.csv_path = os.path.join(save_dir, f"efficiency_{model_name}_{run_id}.csv")
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.headers = [
            'task_idx', 'epoch', 'batch_idx', 
            'train_time_ms', 'peak_mem_gb', 
            'flops_gflops', 'params_m'
        ]
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        
        self.start_time = None
        self.current_flops = None
        self.current_params = None

    def record_model_stats(self, model: torch.nn.Module, input_size=(1, 3, 32, 32)):
        """Calculates FLOPs and Parameters using thop."""
        try:
            from thop import profile
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size).to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            self.current_flops = flops / 1e9  # GFLOPs
            self.current_params = params / 1e6  # M
        except ImportError:
            print("thop not installed. FLOPs/Params recording skipped.")
            self.current_flops = 0
            self.current_params = sum(p.numel() for p in model.parameters()) / 1e6

    def start_timer(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def end_timer_and_record(self, task_idx: int, epoch: int, batch_idx: int):
        if self.start_time is None:
            return
            
        torch.cuda.synchronize()
        elapsed_time_ms = (time.perf_counter() - self.start_time) * 1000
        self.start_time = None
        
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                task_idx, epoch, batch_idx,
                f"{elapsed_time_ms:.2f}",
                f"{peak_mem_gb:.4f}",
                f"{self.current_flops:.4f}" if self.current_flops else "N/A",
                f"{self.current_params:.4f}" if self.current_params else "N/A"
            ])

    @staticmethod
    def reset_peak_memory():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
