# CIFAR10 全部实验参数与结果

## 论文指标

| 实验 | 论文 Class-IL | 论文 Class-IL 区间 | 论文 Task-IL |
|---|---:|---|---|
| CIFAR10 Symmetric 20% | 67.64 ± 0.52 | 67.12 ~ 68.16 | 未记录 |
| CIFAR10 Symmetric 40% | 64.46 ± 0.40 | 64.06 ~ 64.86 | 未记录 |
| CIFAR10 Symmetric 60% | 47.66 ± 0.59 | 47.07 ~ 48.25 | 未记录 |
| CIFAR10 Asymmetric 40% | 57.60 ± 0.70 | 56.90 ~ 58.30 | 未记录 |

## 公共基础参数

| 参数 | 值 |
|---|---|
| dataset | seq-cifar10 |
| model | ogc-sap |
| backbone | resnet18 |
| n_epochs | 50 |
| batch_size | 32 |
| lr | 0.03 |
| buffer_size | 500 |
| seed | 0 |
| savecheck | last |

## CIFAR10 Symm20 参数与结果

| 序号 | 实验来源 | 运行名 / 结果目录 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL | 相对论文 Class-IL 均值 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 本机第 1 次 | results_cifar10_seed0 | symm | 0.2 | 1000 | 0.30 | 0.50 | 1.5 | 1000 | 0 | 63.45 | 90.19 | -4.19 |
| 2 | 本机第 2 次 | results_cifar10_symm20_rerun | symm | 0.2 | 1000 | 0.30 | 0.30 | 2.0 | 400 | 0 | 60.47 | 89.93 | -7.17 |
| 3 | 本机第 3 次 | results_cifar10_symm20_sap3000_lcw05_pen15 | symm | 0.2 | 3000 | 0.30 | 0.50 | 1.5 | 400 | 0 | 62.56 | 90.54 | -5.08 |
| 4 | 本机第 4 次 | results_cifar10_symm20_sap3000_ogc04_lcw02_pen3 | symm | 0.2 | 3000 | 0.40 | 0.20 | 3.0 | 400 | 4 | 65.13 | 90.06 | -2.51 |
| 5 | 本机第 5 次 | results_cifar10_symm20_sap5000_ogc04_lcw02_pen3 | symm | 0.2 | 5000 | 0.40 | 0.20 | 3.0 | 400 | 4 | 65.13 | 90.06 | -2.51 |
| 6 | 远程第 1 次重跑 | cifar10_symm20_exp1_numw4_rerun | symm | 0.2 | 1000 | 0.30 | 0.50 | 1.5 | 1000 | 4 | 64.12 | 89.21 | -3.52 |
| 7 | 远程第 2 次重跑 | cifar10_symm20_exp2_numw4_rerun | symm | 0.2 | 1000 | 0.30 | 0.30 | 2.0 | 400 | 4 | 66.81 | 91.20 | -0.83 |
| 8 | 远程第 3 次重跑 | cifar10_symm20_exp3_numw4_rerun | symm | 0.2 | 3000 | 0.30 | 0.50 | 1.5 | 400 | 4 | 61.43 | 89.06 | -6.21 |

## CIFAR10 Symm20 Raw Accuracy

| 序号 | Class-IL raw accuracies | Task-IL raw accuracies |
|---:|---|---|
| 1 | [62.75, 51.80, 58.30, 70.35, 74.05] | [91.25, 80.05, 89.40, 94.85, 95.40] |
| 2 | [69.90, 48.75, 51.50, 80.45, 51.75] | [91.10, 82.70, 84.90, 95.40, 95.55] |
| 3 | [69.35, 64.00, 50.55, 67.50, 61.40] | [92.40, 84.55, 87.25, 93.40, 95.10] |
| 4 | [66.60, 59.75, 55.30, 69.80, 74.20] | [87.70, 84.05, 88.05, 94.45, 96.05] |
| 5 | [66.60, 59.75, 55.30, 69.80, 74.20] | [87.70, 84.05, 88.05, 94.45, 96.05] |
| 6 | [60.60, 45.85, 55.65, 82.10, 76.40] | [86.70, 80.45, 87.70, 95.25, 95.95] |
| 7 | [70.10, 50.20, 62.00, 73.65, 78.10] | [91.45, 82.20, 90.70, 94.65, 97.00] |
| 8 | [63.80, 54.50, 64.10, 69.90, 54.85] | [90.10, 78.65, 89.45, 94.65, 92.45] |

## CIFAR10 Symm40 前置实验参数与结果

| 组别 | 实验来源 | 运行名 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL | 相对论文 Class-IL 均值 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 原文对齐组 | cifar10_symm40_sap2000_seed0 | symm | 0.4 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 58.21 | 87.74 | -6.25 |
| A | 调参组 A | cifar10_symm40_sap3000_ogc04_lcw025_pen2 | symm | 0.4 | 3000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 60.74 | 88.28 | -3.72 |
| B | 调参组 B | cifar10_symm40_sap5000_ogc04_lcw025_pen2 | symm | 0.4 | 5000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 57.99 | 87.25 | -6.47 |
| C | 调参组 C | cifar10_symm40_sap3000_ogc045_lcw02_pen25 | symm | 0.4 | 3000 | 0.45 | 0.20 | 2.5 | 400 | 4 | 56.63 | 86.13 | -7.83 |

## CIFAR10 Symm40 前置实验 Raw Accuracy

| 组别 | Class-IL raw accuracies | Task-IL raw accuracies |
|---|---|---|
| D | [59.40, 49.10, 52.35, 66.55, 63.65] | [85.90, 80.15, 87.30, 92.50, 92.85] |
| A | [64.05, 55.90, 47.25, 71.80, 64.70] | [89.40, 81.65, 85.00, 91.35, 94.00] |
| B | [65.35, 56.50, 43.65, 59.95, 64.50] | [87.85, 79.35, 82.85, 91.60, 94.60] |
| C | [57.85, 37.00, 58.80, 71.55, 57.95] | [88.00, 77.45, 85.95, 92.40, 86.85] |

## CIFAR10 Symm40 SAP Grid Search 参数与结果

| 序号 | 日志 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL | 相对论文 Class-IL 均值 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | run_logs/cifar10_symm40_sap1000_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 1000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 58.47 | 87.56 | -5.99 |
| 2 | run_logs/cifar10_symm40_sap1500_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 1500 | 0.40 | 0.25 | 2.0 | 400 | 4 | 59.78 | 87.23 | -4.68 |
| 3 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 60.56 | 87.44 | -3.90 |
| 4 | run_logs/cifar10_symm40_sap2500_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2500 | 0.40 | 0.25 | 2.0 | 400 | 4 | 55.60 | 86.16 | -8.86 |
| 5 | run_logs/cifar10_symm40_sap3000_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 3000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 60.54 | 88.09 | -3.92 |
| 6 | run_logs/cifar10_symm40_sap3500_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 3500 | 0.40 | 0.25 | 2.0 | 400 | 4 | 58.77 | 87.55 | -5.69 |
| 7 | run_logs/cifar10_symm40_sap4000_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 4000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 54.23 | 85.24 | -10.23 |
| 8 | run_logs/cifar10_symm40_sap4500_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 4500 | 0.40 | 0.25 | 2.0 | 400 | 4 | 56.13 | 84.41 | -8.33 |
| 9 | run_logs/cifar10_symm40_sap5000_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 5000 | 0.40 | 0.25 | 2.0 | 400 | 4 | 59.37 | 87.09 | -5.09 |

## CIFAR10 Symm40 SAP Grid Search Raw Accuracy

| sap_scale_coff | Class-IL raw accuracies | Task-IL raw accuracies |
|---:|---|---|
| 1000 | [65.30, 47.75, 53.70, 67.25, 58.35] | [88.85, 80.25, 84.10, 91.00, 93.60] |
| 1500 | [67.65, 51.55, 48.30, 65.60, 65.80] | [88.75, 75.40, 85.75, 93.20, 93.05] |
| 2000 | [66.45, 51.95, 52.85, 63.55, 68.00] | [87.40, 79.85, 85.15, 90.20, 94.60] |
| 2500 | [57.00, 43.30, 49.40, 69.65, 58.65] | [89.70, 80.55, 81.30, 90.15, 89.10] |
| 3000 | [60.25, 50.75, 57.10, 62.95, 71.65] | [89.25, 80.20, 86.25, 93.25, 91.50] |
| 3500 | [64.30, 46.35, 52.40, 63.80, 67.00] | [90.90, 77.50, 84.35, 92.05, 92.95] |
| 4000 | [54.45, 58.55, 46.45, 59.85, 51.85] | [80.50, 79.75, 86.85, 90.60, 88.50] |
| 4500 | [66.15, 53.45, 51.65, 58.80, 50.60] | [85.30, 77.30, 82.35, 91.35, 85.75] |
| 5000 | [67.50, 49.65, 51.65, 69.70, 58.35] | [85.75, 78.65, 85.15, 92.30, 93.60] |

## CIFAR10 Symm40 OGC Loss Weight Grid Search 并行结果

| 序号 | 日志 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL | status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | run_logs/cifar10_symm40_sap2000_ogc01_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.1 | 0.25 | 2.0 | 400 | 4 | 57.60 | 86.57 | DONE |
| 2 | run_logs/cifar10_symm40_sap2000_ogc02_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.2 | 0.25 | 2.0 | 400 | 4 | 58.41 | 86.15 | DONE |
| 3 | run_logs/cifar10_symm40_sap2000_ogc03_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.3 | 0.25 | 2.0 | 400 | 4 | 57.89 | 87.93 | DONE |
| 4 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.4 | 0.25 | 2.0 | 400 | 4 | 59.68 | 85.78 | DONE |
| 5 | run_logs/cifar10_symm40_sap2000_ogc05_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.5 | 0.25 | 2.0 | 400 | 4 | 56.46 | 87.20 | DONE |
| 6 | run_logs/cifar10_symm40_sap2000_ogc06_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.6 | 0.25 | 2.0 | 400 | 4 | 56.70 | 87.90 | DONE |
| 7 | run_logs/cifar10_symm40_sap2000_ogc07_lcw025_pen2_seed0_grid.log | symm | 0.4 | 2000 | 0.7 | 0.25 | 2.0 | 400 | 4 | 58.58 | 86.68 | DONE |
| 8 | run_logs/cifar10_symm40_sap3000_ogc01_lcw025_pen2_seed0_grid.log | symm | 0.4 | 3000 | 0.1 | 0.25 | 2.0 | 400 | 4 | 55.76 | 84.35 | DONE |
| 9 | run_logs/cifar10_symm40_sap3000_ogc02_lcw025_pen2_seed0_grid.log | symm | 0.4 | 3000 | 0.2 | 0.25 | 2.0 | 400 | 4 | 59.64 | 88.48 | DONE |
| 10 | run_logs/cifar10_symm40_sap3000_ogc03_lcw025_pen2_seed0_grid.log | symm | 0.4 | 3000 | 0.3 | 0.25 | 2.0 | 400 | 4 | 55.82 | 87.04 | DONE |

## CIFAR10 Symm40 OGC Loss Weight Grid Search 并行 Raw Accuracy

| sap_scale_coff | ogc_loss_weight | Class-IL raw accuracies | Task-IL raw accuracies |
|---:|---:|---|---|
| 2000 | 0.1 | [58.35, 44.80, 53.90, 71.95, 59.00] | [83.60, 80.45, 81.60, 94.20, 93.00] |
| 2000 | 0.2 | [63.60, 46.40, 54.55, 63.10, 64.40] | [87.15, 77.40, 84.20, 91.30, 90.70] |
| 2000 | 0.3 | [49.80, 48.20, 51.45, 68.80, 71.20] | [85.85, 81.30, 86.25, 92.70, 93.55] |
| 2000 | 0.4 | [53.85, 50.00, 52.45, 73.30, 68.80] | [79.85, 77.75, 85.30, 93.15, 92.85] |
| 2000 | 0.5 | [65.65, 44.40, 59.65, 62.45, 50.15] | [84.40, 78.20, 87.85, 92.25, 93.30] |
| 2000 | 0.6 | [61.65, 45.80, 54.85, 65.00, 56.20] | [88.10, 80.35, 85.25, 93.15, 92.65] |
| 2000 | 0.7 | [60.60, 56.65, 41.30, 67.80, 66.55] | [86.80, 78.75, 83.10, 90.65, 94.10] |
| 3000 | 0.1 | [62.25, 57.15, 39.20, 60.75, 59.45] | [88.25, 80.60, 80.15, 87.90, 84.85] |
| 3000 | 0.2 | [62.30, 56.35, 45.75, 67.65, 66.15] | [86.30, 82.70, 85.00, 93.45, 94.95] |
| 3000 | 0.3 | [61.85, 52.55, 51.55, 67.75, 45.40] | [85.50, 80.35, 87.30, 90.00, 92.05] |

## CIFAR10 Symm40 OGC Loss Weight Grid Search 串行 sap=2000 参数与结果

| 序号 | 日志 | 完成时间 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL | 相对论文 Class-IL 均值 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | run_logs/cifar10_symm40_sap2000_ogc01_lcw025_pen2_seed0_serial.log | 2026-07-02 14:02:53 | symm | 0.4 | 2000 | 0.1 | 0.25 | 2.0 | 400 | 4 | 60.37 | 88.90 | -4.09 |
| 2 | run_logs/cifar10_symm40_sap2000_ogc02_lcw025_pen2_seed0_serial.log | 2026-07-02 16:25:30 | symm | 0.4 | 2000 | 0.2 | 0.25 | 2.0 | 400 | 4 | 55.82 | 87.39 | -8.64 |
| 3 | run_logs/cifar10_symm40_sap2000_ogc03_lcw025_pen2_seed0_serial.log | 2026-07-02 18:48:56 | symm | 0.4 | 2000 | 0.3 | 0.25 | 2.0 | 400 | 4 | 57.12 | 86.16 | -7.34 |
| 4 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log | 2026-07-02 21:11:58 | symm | 0.4 | 2000 | 0.4 | 0.25 | 2.0 | 400 | 4 | 57.56 | 86.39 | -6.90 |
| 5 | run_logs/cifar10_symm40_sap2000_ogc05_lcw025_pen2_seed0_serial.log | 2026-07-02 23:35:46 | symm | 0.4 | 2000 | 0.5 | 0.25 | 2.0 | 400 | 4 | 59.42 | 87.10 | -5.04 |
| 6 | run_logs/cifar10_symm40_sap2000_ogc06_lcw025_pen2_seed0_serial.log | 2026-07-03 01:59:23 | symm | 0.4 | 2000 | 0.6 | 0.25 | 2.0 | 400 | 4 | 57.68 | 87.21 | -6.78 |
| 7 | run_logs/cifar10_symm40_sap2000_ogc07_lcw025_pen2_seed0_serial.log | 2026-07-03 04:23:57 | symm | 0.4 | 2000 | 0.7 | 0.25 | 2.0 | 400 | 4 | 59.52 | 87.37 | -4.94 |

## CIFAR10 Symm40 ogc=0.4 同超参数实验结果

| 序号 | 实验来源 | 运行名 / 日志 | 时间记录 | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | Class-IL | Task-IL |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 原文对齐组 | cifar10_symm40_sap2000_seed0 | 记录文件时间：2026-06-27 23:29:13 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 58.21 | 87.74 |
| 2 | sap_scale_coff grid search | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 记录文件时间：2026-06-30 16:19:22 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 60.56 | 87.44 |
| 3 | ogc_loss_weight grid search 并行运行 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 记录文件时间：2026-07-02 10:58:36 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 59.68 | 85.78 |
| 4 | ogc_loss_weight grid search 串行重跑 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log | 2026-07-02 21:11:58 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 57.56 | 86.39 |

## CIFAR10 Symm40 ogc=0.4 同超参数 Raw Accuracy

| 实验来源 | Class-IL raw accuracies | Task-IL raw accuracies |
|---|---|---|
| 原文对齐组 | [59.40, 49.10, 52.35, 66.55, 63.65] | [85.90, 80.15, 87.30, 92.50, 92.85] |
| sap_scale_coff grid search | [66.45, 51.95, 52.85, 63.55, 68.00] | [87.40, 79.85, 85.15, 90.20, 94.60] |
| ogc_loss_weight grid search 并行运行 | [53.85, 50.00, 52.45, 73.30, 68.80] | [79.85, 77.75, 85.30, 93.15, 92.85] |

## CIFAR10 Symm40 串行 ogc=0.4 逐任务累计结果

| task 数 | 时间 | Class-IL | Task-IL |
|---:|---|---:|---:|
| 1 | 2026-07-02 19:16:52 | 94.95 | 94.95 |
| 2 | 2026-07-02 19:44:58 | 77.28 | 86.65 |
| 3 | 2026-07-02 20:13:30 | 68.23 | 87.12 |
| 4 | 2026-07-02 20:42:24 | 61.42 | 86.90 |
| 5 | 2026-07-02 21:11:58 | 57.56 | 86.39 |

## CIFAR10 Symm60 参数与结果

| 组别 | 实验来源 | 运行名 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL | 相对论文 Class-IL 均值 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 原文对齐组 | cifar10_symm60_sap3000_seed0 | symm | 0.6 | 3000 | 0.50 | 0.20 | 2.0 | 400 | 4 | 45.05 | 79.82 | -2.61 |
| A | 调参组 A | cifar10_symm60_sap5000_ogc05_lcw02_pen2 | symm | 0.6 | 5000 | 0.50 | 0.20 | 2.0 | 400 | 4 | 44.54 | 80.32 | -3.12 |
| B | 调参组 B | cifar10_symm60_sap5000_ogc055_lcw015_pen25 | symm | 0.6 | 5000 | 0.55 | 0.15 | 2.5 | 400 | 4 | 45.27 | 80.93 | -2.39 |
| C | 调参组 C | cifar10_symm60_sap3000_ogc055_lcw015_pen3 | symm | 0.6 | 3000 | 0.55 | 0.15 | 3.0 | 400 | 4 | 44.38 | 81.13 | -3.28 |

## CIFAR10 Symm60 Raw Accuracy

| 组别 | Class-IL raw accuracies | Task-IL raw accuracies |
|---|---|---|
| D | [55.15, 26.75, 35.00, 55.70, 52.65] | [85.10, 64.35, 74.20, 88.05, 87.40] |
| A | [48.05, 35.25, 30.95, 53.45, 55.00] | [85.65, 71.00, 76.95, 85.70, 82.30] |
| B | [43.00, 26.75, 39.70, 50.10, 66.80] | [82.05, 73.95, 76.15, 86.10, 86.40] |
| C | [50.35, 30.10, 36.50, 56.90, 48.05] | [82.20, 72.60, 79.35, 85.05, 86.45] |

## CIFAR10 Asym40 参数与结果

| 组别 | 实验来源 | 运行名 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL | 相对论文 Class-IL 均值 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 推荐起点组 | cifar10_asym40_A_sap700_ogc04_lcw03_pen2 | asym | 0.4 | 700 | 0.40 | 0.30 | 2.0 | 400 | 4 | 52.16 | 89.69 | -5.44 |
| B | SAP 增强组 | cifar10_asym40_B_sap1000_ogc04_lcw03_pen2 | asym | 0.4 | 1000 | 0.40 | 0.30 | 2.0 | 400 | 4 | 56.96 | 91.27 | -0.64 |
| C | SAP + OGC/buffer 增强组 | cifar10_asym40_C_sap1000_ogc045_lcw025_pen25 | asym | 0.4 | 1000 | 0.45 | 0.25 | 2.5 | 400 | 4 | 56.75 | 91.87 | -0.85 |

## CIFAR10 Asym40 Raw Accuracy

| 组别 | Class-IL raw accuracies | Task-IL raw accuracies |
|---|---|---|
| A | [59.90, 39.45, 29.75, 70.70, 61.00] | [91.30, 77.30, 87.80, 95.05, 97.00] |
| B | [70.60, 38.45, 42.30, 79.00, 54.45] | [93.95, 81.30, 88.00, 96.70, 96.40] |
| C | [69.20, 44.60, 39.15, 79.05, 51.75] | [90.60, 84.65, 88.45, 97.40, 98.25] |

## 已有参数但无本地结果记录的 CIFAR10 OGC Grid Search 组合

| 序号 | 运行顺序 | 日志 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | num_workers | Class-IL | Task-IL |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 11 | run_logs/cifar10_symm40_sap3000_ogc04_lcw025_pen2_seed0_serial.log | symm | 0.4 | 3000 | 0.4 | 0.25 | 2.0 | 400 | 4 | 未记录 | 未记录 |
| 2 | 12 | run_logs/cifar10_symm40_sap3000_ogc05_lcw025_pen2_seed0_serial.log | symm | 0.4 | 3000 | 0.5 | 0.25 | 2.0 | 400 | 4 | 未记录 | 未记录 |
| 3 | 13 | run_logs/cifar10_symm40_sap3000_ogc06_lcw025_pen2_seed0_serial.log | symm | 0.4 | 3000 | 0.6 | 0.25 | 2.0 | 400 | 4 | 未记录 | 未记录 |
| 4 | 14 | run_logs/cifar10_symm40_sap3000_ogc07_lcw025_pen2_seed0_serial.log | symm | 0.4 | 3000 | 0.7 | 0.25 | 2.0 | 400 | 4 | 未记录 | 未记录 |
