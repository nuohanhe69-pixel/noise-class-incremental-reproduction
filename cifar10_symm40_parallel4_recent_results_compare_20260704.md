# CIFAR10 Symm40 OGC Grid Search parallel4 最近结果对比

## 论文指标

| 实验 | 论文 Class-IL 均值 | 论文 Class-IL 标准差 | 论文 Class-IL 区间 | 论文 Task-IL |
|---|---:|---:|---|---|
| CIFAR10 Symmetric 40% | 64.46 | 0.40 | 64.06 ~ 64.86 | 未记录 |

## 本次实验公共参数

| 参数 | 值 |
|---|---|
| dataset | seq-cifar10 |
| model | ogc-sap |
| noise_type | symm |
| noise_rate | 0.4 |
| backbone | resnet18 |
| n_epochs | 50 |
| batch_size | 32 |
| lr | 0.03 |
| buffer_size | 500 |
| num_workers | 4 |
| seed | 0 |
| savecheck | last |
| ogc_low_conf_weight | 0.25 |
| ogc_buffer_penalty_coeff | 2.0 |
| sap_retain_samples | 400 |
| 并行方式 | parallel4 |

## 本次实验变化参数

| 参数 | 取值 |
|---|---|
| sap_scale_coff | 2000, 3000 |
| ogc_loss_weight | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 |

## 本次实验参数与结果

| 序号 | 日志 | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | Class-IL | Task-IL |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | run_logs/cifar10_symm40_sap2000_ogc01_lcw025_pen2_seed0_parallel4.log | 2000 | 0.1 | 0.25 | 2.0 | 400 | 61.84 | 88.34 |
| 2 | run_logs/cifar10_symm40_sap2000_ogc02_lcw025_pen2_seed0_parallel4.log | 2000 | 0.2 | 0.25 | 2.0 | 400 | 59.13 | 86.95 |
| 3 | run_logs/cifar10_symm40_sap2000_ogc03_lcw025_pen2_seed0_parallel4.log | 2000 | 0.3 | 0.25 | 2.0 | 400 | 58.89 | 88.91 |
| 4 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_parallel4.log | 2000 | 0.4 | 0.25 | 2.0 | 400 | 59.77 | 88.24 |
| 5 | run_logs/cifar10_symm40_sap2000_ogc05_lcw025_pen2_seed0_parallel4.log | 2000 | 0.5 | 0.25 | 2.0 | 400 | 59.18 | 87.85 |
| 6 | run_logs/cifar10_symm40_sap2000_ogc06_lcw025_pen2_seed0_parallel4.log | 2000 | 0.6 | 0.25 | 2.0 | 400 | 58.86 | 88.04 |
| 7 | run_logs/cifar10_symm40_sap2000_ogc07_lcw025_pen2_seed0_parallel4.log | 2000 | 0.7 | 0.25 | 2.0 | 400 | 59.09 | 88.75 |
| 8 | run_logs/cifar10_symm40_sap3000_ogc01_lcw025_pen2_seed0_parallel4.log | 3000 | 0.1 | 0.25 | 2.0 | 400 | 58.46 | 87.94 |
| 9 | run_logs/cifar10_symm40_sap3000_ogc02_lcw025_pen2_seed0_parallel4.log | 3000 | 0.2 | 0.25 | 2.0 | 400 | 58.37 | 88.48 |
| 10 | run_logs/cifar10_symm40_sap3000_ogc03_lcw025_pen2_seed0_parallel4.log | 3000 | 0.3 | 0.25 | 2.0 | 400 | 62.72 | 88.95 |
| 11 | run_logs/cifar10_symm40_sap3000_ogc04_lcw025_pen2_seed0_parallel4.log | 3000 | 0.4 | 0.25 | 2.0 | 400 | 59.10 | 89.08 |
| 12 | run_logs/cifar10_symm40_sap3000_ogc05_lcw025_pen2_seed0_parallel4.log | 3000 | 0.5 | 0.25 | 2.0 | 400 | 60.93 | 88.43 |
| 13 | run_logs/cifar10_symm40_sap3000_ogc06_lcw025_pen2_seed0_parallel4.log | 3000 | 0.6 | 0.25 | 2.0 | 400 | 61.18 | 89.24 |
| 14 | run_logs/cifar10_symm40_sap3000_ogc07_lcw025_pen2_seed0_parallel4.log | 3000 | 0.7 | 0.25 | 2.0 | 400 | 60.72 | 89.20 |

## 本次实验 Raw Accuracy

| sap_scale_coff | ogc_loss_weight | Class-IL raw accuracies | Task-IL raw accuracies |
|---:|---:|---|---|
| 2000 | 0.1 | [58.10, 45.95, 60.65, 67.90, 76.60] | [85.90, 82.70, 87.20, 94.70, 91.20] |
| 2000 | 0.2 | [56.25, 41.30, 47.35, 76.40, 74.35] | [86.90, 81.30, 80.70, 94.20, 91.65] |
| 2000 | 0.3 | [52.90, 46.75, 51.80, 66.85, 76.15] | [88.75, 82.10, 89.20, 91.95, 92.55] |
| 2000 | 0.4 | [62.05, 44.15, 47.20, 74.60, 70.85] | [88.40, 79.90, 86.05, 92.10, 94.75] |
| 2000 | 0.5 | [57.50, 43.20, 60.00, 59.55, 75.65] | [89.15, 78.65, 83.25, 92.85, 95.35] |
| 2000 | 0.6 | [54.70, 43.85, 58.85, 59.35, 77.55] | [88.75, 77.15, 86.25, 92.00, 96.05] |
| 2000 | 0.7 | [55.50, 48.00, 59.60, 61.95, 70.40] | [89.50, 80.35, 86.75, 93.05, 94.10] |
| 3000 | 0.1 | [56.55, 50.10, 54.40, 58.50, 72.75] | [90.95, 76.80, 87.40, 93.50, 91.05] |
| 3000 | 0.2 | [56.45, 49.15, 54.55, 65.60, 66.10] | [88.90, 81.20, 88.75, 90.00, 93.55] |
| 3000 | 0.3 | [67.35, 50.20, 53.25, 67.15, 75.65] | [90.20, 80.15, 85.95, 93.05, 95.40] |
| 3000 | 0.4 | [56.20, 46.80, 44.35, 72.05, 76.10] | [90.75, 81.75, 85.00, 93.15, 94.75] |
| 3000 | 0.5 | [61.30, 52.40, 51.85, 70.25, 68.85] | [88.95, 81.35, 85.45, 93.05, 93.35] |
| 3000 | 0.6 | [59.00, 45.15, 52.60, 67.85, 81.30] | [90.35, 80.85, 87.60, 93.55, 93.85] |
| 3000 | 0.7 | [63.00, 47.15, 49.45, 71.45, 72.55] | [92.10, 78.20, 87.35, 93.30, 95.05] |

## 本次实验最优结果

| 对比项 | sap_scale_coff | ogc_loss_weight | Class-IL | Task-IL | ΔClass-IL vs 原文对齐组 | ΔTask-IL vs 原文对齐组 | ΔClass-IL vs 论文均值 | ΔClass-IL vs 论文下界 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Class-IL 最高 | 3000 | 0.3 | 62.72 | 88.95 | +4.51 | +1.21 | -1.74 | -1.34 |
| Task-IL 最高 | 3000 | 0.6 | 61.18 | 89.24 | +2.97 | +1.50 | -3.28 | -2.88 |
