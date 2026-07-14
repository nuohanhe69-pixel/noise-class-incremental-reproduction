# CIFAR10 Symm40 sap=2000 OGC Grid Search 串行结果对比

## 论文指标

| 实验 | Class-IL 均值 | Class-IL 标准差 | Class-IL 区间 | Task-IL |
|---|---:|---:|---|---|
| CIFAR10 Symmetric 40% | 64.46 | 0.40 | 64.06 ~ 64.86 | 未记录 |

## 公共实验参数

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
| sap_scale_coff | 2000 |
| ogc_low_conf_weight | 0.25 |
| ogc_buffer_penalty_coeff | 2.0 |
| sap_retain_samples | 400 |

## sap=2000 串行实验结果

| 组别 | 日志 | 完成时间 | sap_scale_coff | ogc_loss_weight | Class-IL | Task-IL |
|---:|---|---|---:|---:|---:|---:|
| 1 | run_logs/cifar10_symm40_sap2000_ogc01_lcw025_pen2_seed0_serial.log | 2026-07-02 14:02:53 | 2000 | 0.1 | 60.37 | 88.90 |
| 2 | run_logs/cifar10_symm40_sap2000_ogc02_lcw025_pen2_seed0_serial.log | 2026-07-02 16:25:30 | 2000 | 0.2 | 55.82 | 87.39 |
| 3 | run_logs/cifar10_symm40_sap2000_ogc03_lcw025_pen2_seed0_serial.log | 2026-07-02 18:48:56 | 2000 | 0.3 | 57.12 | 86.16 |
| 4 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log | 2026-07-02 21:11:58 | 2000 | 0.4 | 57.56 | 86.39 |
| 5 | run_logs/cifar10_symm40_sap2000_ogc05_lcw025_pen2_seed0_serial.log | 2026-07-02 23:35:46 | 2000 | 0.5 | 59.42 | 87.10 |
| 6 | run_logs/cifar10_symm40_sap2000_ogc06_lcw025_pen2_seed0_serial.log | 2026-07-03 01:59:23 | 2000 | 0.6 | 57.68 | 87.21 |
| 7 | run_logs/cifar10_symm40_sap2000_ogc07_lcw025_pen2_seed0_serial.log | 2026-07-03 04:23:57 | 2000 | 0.7 | 59.52 | 87.37 |

## sap=2000 串行实验与论文 Class-IL 指标对比

| ogc_loss_weight | Class-IL | 论文 Class-IL 均值 | ΔClass-IL 均值 | 论文 Class-IL 下界 | ΔClass-IL 下界 | 论文 Class-IL 上界 | ΔClass-IL 上界 | Task-IL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 60.37 | 64.46 | -4.09 | 64.06 | -3.69 | 64.86 | -4.49 | 88.90 |
| 0.2 | 55.82 | 64.46 | -8.64 | 64.06 | -8.24 | 64.86 | -9.04 | 87.39 |
| 0.3 | 57.12 | 64.46 | -7.34 | 64.06 | -6.94 | 64.86 | -7.74 | 86.16 |
| 0.4 | 57.56 | 64.46 | -6.90 | 64.06 | -6.50 | 64.86 | -7.30 | 86.39 |
| 0.5 | 59.42 | 64.46 | -5.04 | 64.06 | -4.64 | 64.86 | -5.44 | 87.10 |
| 0.6 | 57.68 | 64.46 | -6.78 | 64.06 | -6.38 | 64.86 | -7.18 | 87.21 |
| 0.7 | 59.52 | 64.46 | -4.94 | 64.06 | -4.54 | 64.86 | -5.34 | 87.37 |

## ogc04 同超参数实验清单

| 序号 | 实验来源 | 运行名 / 日志 | 时间记录 | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 原文对齐组 | cifar10_symm40_sap2000_seed0 | 记录文件时间：2026-06-27 23:29:13 | 2000 | 0.40 | 0.25 | 2.0 | 400 |
| 2 | sap_scale_coff grid search | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 记录文件时间：2026-06-30 16:19:22 | 2000 | 0.40 | 0.25 | 2.0 | 400 |
| 3 | ogc_loss_weight grid search 并行运行 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 记录文件时间：2026-07-02 10:58:36 | 2000 | 0.40 | 0.25 | 2.0 | 400 |
| 4 | ogc_loss_weight grid search 串行重跑 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log | 完成时间：2026-07-02 21:11:58 | 2000 | 0.40 | 0.25 | 2.0 | 400 |

## ogc04 同超参数实验参数对比

| 参数 | 原文对齐组 | sap_scale_coff grid search | ogc grid search 并行运行 | ogc grid search 串行重跑 |
|---|---|---|---|---|
| dataset | seq-cifar10 | seq-cifar10 | seq-cifar10 | seq-cifar10 |
| model | ogc-sap | ogc-sap | ogc-sap | ogc-sap |
| noise_type | symm | symm | symm | symm |
| noise_rate | 0.4 | 0.4 | 0.4 | 0.4 |
| backbone | resnet18 | resnet18 | resnet18 | resnet18 |
| n_epochs | 50 | 50 | 50 | 50 |
| batch_size | 32 | 32 | 32 | 32 |
| lr | 0.03 | 0.03 | 0.03 | 0.03 |
| buffer_size | 500 | 500 | 500 | 500 |
| num_workers | 4 | 4 | 4 | 4 |
| seed | 0 | 0 | 0 | 0 |
| savecheck | last | last | last | last |
| sap_scale_coff | 2000 | 2000 | 2000 | 2000 |
| ogc_loss_weight | 0.40 | 0.40 | 0.40 | 0.40 |
| ogc_low_conf_weight | 0.25 | 0.25 | 0.25 | 0.25 |
| ogc_buffer_penalty_coeff | 2.0 | 2.0 | 2.0 | 2.0 |
| sap_retain_samples | 400 | 400 | 400 | 400 |

## ogc04 同超参数实验结果对比

| 实验来源 | 运行名 / 日志 | Class-IL | Task-IL | 相对当前串行 ΔClass-IL | 相对当前串行 ΔTask-IL | 相对论文均值 ΔClass-IL | 相对论文下界 ΔClass-IL |
|---|---|---:|---:|---:|---:|---:|---:|
| 原文对齐组 | cifar10_symm40_sap2000_seed0 | 58.21 | 87.74 | +0.65 | +1.35 | -6.25 | -5.85 |
| sap_scale_coff grid search | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 60.56 | 87.44 | +3.00 | +1.05 | -3.90 | -3.50 |
| ogc_loss_weight grid search 并行运行 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 59.68 | 85.78 | +2.12 | -0.61 | -4.78 | -4.38 |
| ogc_loss_weight grid search 串行重跑 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log | 57.56 | 86.39 | 0.00 | 0.00 | -6.90 | -6.50 |

## ogc04 同超参数历史 Raw Accuracy

| 实验来源 | Class-IL raw accuracies | Task-IL raw accuracies |
|---|---|---|
| 原文对齐组 | [59.40, 49.10, 52.35, 66.55, 63.65] | [85.90, 80.15, 87.30, 92.50, 92.85] |
| sap_scale_coff grid search | [66.45, 51.95, 52.85, 63.55, 68.00] | [87.40, 79.85, 85.15, 90.20, 94.60] |
| ogc_loss_weight grid search 并行运行 | [53.85, 50.00, 52.45, 73.30, 68.80] | [79.85, 77.75, 85.30, 93.15, 92.85] |
