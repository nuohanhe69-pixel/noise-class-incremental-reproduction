# CIFAR10 Symm20 实验统计报告

## 正式实验汇总

| 序号 | 结果目录 | Class-IL accmean_task5 | Task-IL accmean_task5 | Class-IL raw accuracies | Task-IL raw accuracies | 与论文均值差距 |
|---:|---|---:|---:|---|---|---:|
| 1 | `results_cifar10_seed0` | 63.45 | 90.19 | `[62.75, 51.80, 58.30, 70.35, 74.05]` | `[91.25, 80.05, 89.40, 94.85, 95.40]` | -4.19 |
| 2 | `results_cifar10_symm20_rerun` | 60.47 | 89.93 | `[69.90, 48.75, 51.50, 80.45, 51.75]` | `[91.10, 82.70, 84.90, 95.40, 95.55]` | -7.17 |
| 3 | `results_cifar10_symm20_sap3000_lcw05_pen15` | 62.56 | 90.54 | `[69.35, 64.00, 50.55, 67.50, 61.40]` | `[92.40, 84.55, 87.25, 93.40, 95.10]` | -5.08 |
| 4 | `results_cifar10_symm20_sap3000_ogc04_lcw02_pen3` | 65.13 | 90.06 | `[66.60, 59.75, 55.30, 69.80, 74.20]` | `[87.70, 84.05, 88.05, 94.45, 96.05]` | -2.51 |
| 5 | `results_cifar10_symm20_sap5000_ogc04_lcw02_pen3` | 65.13 | 90.06 | `[66.60, 59.75, 55.30, 69.80, 74.20]` | `[87.70, 84.05, 88.05, 94.45, 96.05]` | -2.51 |

目前正式实验中最好的 `Class-IL accmean_task5` 为 `65.13`，出现在第 4 次和第 5 次实验。相比论文均值 `67.64` 低 `2.51` 个百分点；相比论文下界 `67.12` 低 `1.99` 个百分点。

## 共同基础参数

5 次正式实验的共同基础设置如下：

| 参数 | 值 |
|---|---|
| `dataset` | `seq-cifar10` |
| `model` | `ogc-sap` |
| `noise_rate` | `0.2` |
| `noise_type` | `symmetric`，对应命令中的 `symm` |
| `seed` | `0` |
| `backbone` | `resnet18` |
| `n_epochs` | `50` |
| `batch_size` | `32` |
| `lr` | `0.03` |
| `buffer_size` | `500` |
| `savecheck` | `last` |

## 每次正式实验参数与结果

### 1. `results_cifar10_seed0`

参数：

| 参数 | 值 |
|---|---|
| `sap_scale_coff` | `[1000.0]` |
| `ogc_loss_weight` | `0.3` |
| `ogc_low_conf_weight` | `0.5` |
| `ogc_buffer_penalty_coeff` | `1.5` |
| `sap_retain_samples` | `1000` |
| `num_workers` | `0` |
| `ogc_run_id` | `cifar10_symm20_seed0` |
| `results_path` | `results_cifar10_seed0/` |
| `checkpoint_path` | `checkpoints/cifar10_seed0/` |
| `run_log` | `run_logs/cifar10_symm20_seed0.log` |

结果：

| 指标 | 值 |
|---|---|
| `Class-IL accmean_task5` | `63.45` |
| `Class-IL raw accuracies` | `[62.75, 51.80, 58.30, 70.35, 74.05]` |
| `Task-IL accmean_task5` | `90.19` |
| `Task-IL raw accuracies` | `[91.25, 80.05, 89.40, 94.85, 95.40]` |

### 2. `results_cifar10_symm20_rerun`

参数：

| 参数 | 值 |
|---|---|
| `sap_scale_coff` | `[1000.0]` |
| `ogc_loss_weight` | `0.3` |
| `ogc_low_conf_weight` | `0.3` |
| `ogc_buffer_penalty_coeff` | `2.0` |
| `sap_retain_samples` | `400` |
| `num_workers` | `0` |
| `ogc_run_id` | `cifar10_symm20_seed0_rerun` |
| `results_path` | `results_cifar10_symm20_rerun/` |
| `checkpoint_path` | `checkpoints/cifar10_symm20_rerun/` |
| `run_log` | `run_logs/cifar10_symm20_seed0_rerun.log` |

结果：

| 指标 | 值 |
|---|---|
| `Class-IL accmean_task5` | `60.47` |
| `Class-IL raw accuracies` | `[69.90, 48.75, 51.50, 80.45, 51.75]` |
| `Task-IL accmean_task5` | `89.93` |
| `Task-IL raw accuracies` | `[91.10, 82.70, 84.90, 95.40, 95.55]` |

### 3. `results_cifar10_symm20_sap3000_lcw05_pen15`

参数：

| 参数 | 值 |
|---|---|
| `sap_scale_coff` | `[3000.0]` |
| `ogc_loss_weight` | `0.3` |
| `ogc_low_conf_weight` | `0.5` |
| `ogc_buffer_penalty_coeff` | `1.5` |
| `sap_retain_samples` | `400` |
| `num_workers` | `0` |
| `ogc_run_id` | `cifar10_symm20_seed0` |
| `results_path` | `results_cifar10_symm20_sap3000_lcw05_pen15/` |
| `checkpoint_path` | `checkpoints/cifar10_symm20_sap3000_lcw05_pen15/` |
| `run_log` | `run_logs/cifar10_symm20_seed0_sap3000_lcw05_pen15.log` |

结果：

| 指标 | 值 |
|---|---|
| `Class-IL accmean_task5` | `62.56` |
| `Class-IL raw accuracies` | `[69.35, 64.00, 50.55, 67.50, 61.40]` |
| `Task-IL accmean_task5` | `90.54` |
| `Task-IL raw accuracies` | `[92.40, 84.55, 87.25, 93.40, 95.10]` |

### 4. `results_cifar10_symm20_sap3000_ogc04_lcw02_pen3`

参数：

| 参数 | 值 |
|---|---|
| `sap_scale_coff` | `[3000.0]` |
| `ogc_loss_weight` | `0.4` |
| `ogc_low_conf_weight` | `0.2` |
| `ogc_buffer_penalty_coeff` | `3.0` |
| `sap_retain_samples` | `400` |
| `num_workers` | `4` |
| `ogc_run_id` | `default` |
| `results_path` | `results_cifar10_symm20_sap3000_ogc04_lcw02_pen3/` |
| `checkpoint_path` | `checkpoints/cifar10_symm20_sap3000_ogc04_lcw02_pen3/` |
| `run_log` | `run_logs/cifar10_symm20_seed0_sap3000_ogc04_lcw02_pen3.log` |

结果：

| 指标 | 值 |
|---|---|
| `Class-IL accmean_task5` | `65.13` |
| `Class-IL raw accuracies` | `[66.60, 59.75, 55.30, 69.80, 74.20]` |
| `Task-IL accmean_task5` | `90.06` |
| `Task-IL raw accuracies` | `[87.70, 84.05, 88.05, 94.45, 96.05]` |

### 5. `results_cifar10_symm20_sap5000_ogc04_lcw02_pen3`

参数：

| 参数 | 值 |
|---|---|
| `sap_scale_coff` | `[5000.0]` |
| `ogc_loss_weight` | `0.4` |
| `ogc_low_conf_weight` | `0.2` |
| `ogc_buffer_penalty_coeff` | `3.0` |
| `sap_retain_samples` | `400` |
| `num_workers` | `4` |
| `ogc_run_id` | `default` |
| `results_path` | `results_cifar10_symm20_sap5000_ogc04_lcw02_pen3/` |
| `checkpoint_path` | `checkpoints/cifar10_symm20_sap5000_ogc04_lcw02_pen3/` |
| `run_log` | `run_logs/cifar10_symm20_seed0_sap5000_ogc04_lcw02_pen3.log` |

结果：

| 指标 | 值 |
|---|---|
| `Class-IL accmean_task5` | `65.13` |
| `Class-IL raw accuracies` | `[66.60, 59.75, 55.30, 69.80, 74.20]` |
| `Task-IL accmean_task5` | `90.06` |
| `Task-IL raw accuracies` | `[87.70, 84.05, 88.05, 94.45, 96.05]` |
