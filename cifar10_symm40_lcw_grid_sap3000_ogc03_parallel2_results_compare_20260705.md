# CIFAR10 Symm40 ogc_low_conf_weight Grid Search 结果对比

## 论文结果数据

| 实验 | 方法 | Class-IL | 标准差 | Class-IL 区间 | Task-IL |
|---|---|---:|---:|---|---|
| CIFAR10 Symmetric 40% | Ours | 64.46 | 0.40 | 64.06 ~ 64.86 | 未记录 |
| CIFAR10 Symmetric 40% | AER | 60.87 | 0.98 | 59.89 ~ 61.85 | 未记录 |

## 原文对齐组结果数据

| 实验来源 | 运行名 | Class-IL | Task-IL | Class-IL raw | Task-IL raw |
|---|---|---:|---:|---|---|
| 原文对齐组 | cifar10_symm40_sap2000_seed0 | 58.21 | 87.74 | [59.40, 49.10, 52.35, 66.55, 63.65] | [85.90, 80.15, 87.30, 92.50, 92.85] |

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
| sap_scale_coff | 3000 |
| ogc_loss_weight | 0.3 |
| ogc_buffer_penalty_coeff | 2.0 |
| sap_retain_samples | 400 |
| 并行方式 | parallel2 |
| grid search 参数 | ogc_low_conf_weight |
| grid search 范围 | 0.05 ~ 0.50 |
| grid search 步长 | 0.05 |

## 本次实验与原文对齐组参数对比

| 参数 | 原文对齐组 | 本次实验 | 差值 |
|---|---:|---:|---:|
| sap_scale_coff | 2000 | 3000 | +1000 |
| ogc_loss_weight | 0.40 | 0.30 | -0.10 |
| ogc_low_conf_weight | 0.25 | grid search | grid search |
| ogc_buffer_penalty_coeff | 2.0 | 2.0 | 0.0 |
| sap_retain_samples | 400 | 400 | 0 |
| noise_rate | 0.4 | 0.4 | 0.0 |
| seed | 0 | 0 | 0 |

## 本次实验已完成结果

| ogc_low_conf_weight | 日志 | 完成时间 | Class-IL | Task-IL |
|---:|---|---|---:|---:|
| 0.05 | run_logs/cifar10_symm40_sap3000_ogc03_lcw005_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 12:31:58 | 54.09 | 85.16 |
| 0.10 | run_logs/cifar10_symm40_sap3000_ogc03_lcw010_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 12:29:19 | 57.87 | 87.88 |
| 0.25 | run_logs/cifar10_symm40_sap3000_ogc03_lcw025_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 03:47:26 | 59.11 | 86.88 |
| 0.30 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 03:47:47 | 57.30 | 87.90 |
| 0.35 | run_logs/cifar10_symm40_sap3000_ogc03_lcw035_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 06:41:26 | 60.49 | 86.86 |
| 0.40 | run_logs/cifar10_symm40_sap3000_ogc03_lcw040_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 06:41:20 | 58.49 | 88.18 |
| 0.45 | run_logs/cifar10_symm40_sap3000_ogc03_lcw045_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 09:36:08 | 58.10 | 88.38 |
| 0.50 | run_logs/cifar10_symm40_sap3000_ogc03_lcw050_pen2_seed0_lcwgrid_parallel2.log | 2026-07-05 09:34:16 | 57.97 | 86.51 |

## 本次实验未完成最终指标

| ogc_low_conf_weight | 日志 | 截图中状态 |
|---:|---|---|
| 0.15 | run_logs/cifar10_symm40_sap3000_ogc03_lcw015_pen2_seed0_lcwgrid_parallel2.log | grep -L 显示未包含 Accuracy for 5 task(s) |
| 0.20 | run_logs/cifar10_symm40_sap3000_ogc03_lcw020_pen2_seed0_lcwgrid_parallel2.log | grep -L 显示未包含 Accuracy for 5 task(s) |

## 本次实验结果与论文 Ours 对比

| ogc_low_conf_weight | Class-IL | 论文 Ours Class-IL | ΔClass-IL vs 论文 Ours | ΔClass-IL vs 论文 Ours 下界 | Task-IL |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 54.09 | 64.46 | -10.37 | -9.97 | 85.16 |
| 0.10 | 57.87 | 64.46 | -6.59 | -6.19 | 87.88 |
| 0.25 | 59.11 | 64.46 | -5.35 | -4.95 | 86.88 |
| 0.30 | 57.30 | 64.46 | -7.16 | -6.76 | 87.90 |
| 0.35 | 60.49 | 64.46 | -3.97 | -3.57 | 86.86 |
| 0.40 | 58.49 | 64.46 | -5.97 | -5.57 | 88.18 |
| 0.45 | 58.10 | 64.46 | -6.36 | -5.96 | 88.38 |
| 0.50 | 57.97 | 64.46 | -6.49 | -6.09 | 86.51 |

## 本次实验结果与论文 AER 对比

| ogc_low_conf_weight | Class-IL | 论文 AER Class-IL | ΔClass-IL vs 论文 AER | Task-IL |
|---:|---:|---:|---:|---:|
| 0.05 | 54.09 | 60.87 | -6.78 | 85.16 |
| 0.10 | 57.87 | 60.87 | -3.00 | 87.88 |
| 0.25 | 59.11 | 60.87 | -1.76 | 86.88 |
| 0.30 | 57.30 | 60.87 | -3.57 | 87.90 |
| 0.35 | 60.49 | 60.87 | -0.38 | 86.86 |
| 0.40 | 58.49 | 60.87 | -2.38 | 88.18 |
| 0.45 | 58.10 | 60.87 | -2.77 | 88.38 |
| 0.50 | 57.97 | 60.87 | -2.90 | 86.51 |

## 本次实验结果与原文对齐组对比

| ogc_low_conf_weight | Class-IL | 原文对齐组 Class-IL | ΔClass-IL vs 原文对齐组 | Task-IL | 原文对齐组 Task-IL | ΔTask-IL vs 原文对齐组 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 54.09 | 58.21 | -4.12 | 85.16 | 87.74 | -2.58 |
| 0.10 | 57.87 | 58.21 | -0.34 | 87.88 | 87.74 | +0.14 |
| 0.25 | 59.11 | 58.21 | +0.90 | 86.88 | 87.74 | -0.86 |
| 0.30 | 57.30 | 58.21 | -0.91 | 87.90 | 87.74 | +0.16 |
| 0.35 | 60.49 | 58.21 | +2.28 | 86.86 | 87.74 | -0.88 |
| 0.40 | 58.49 | 58.21 | +0.28 | 88.18 | 87.74 | +0.44 |
| 0.45 | 58.10 | 58.21 | -0.11 | 88.38 | 87.74 | +0.64 |
| 0.50 | 57.97 | 58.21 | -0.24 | 86.51 | 87.74 | -1.23 |

## 本次实验结果排序

| 排序依据 | ogc_low_conf_weight | Class-IL | Task-IL | ΔClass-IL vs 论文 Ours | ΔClass-IL vs 原文对齐组 | ΔTask-IL vs 原文对齐组 |
|---|---:|---:|---:|---:|---:|---:|
| Class-IL 最高 | 0.35 | 60.49 | 86.86 | -3.97 | +2.28 | -0.88 |
| Task-IL 最高 | 0.45 | 58.10 | 88.38 | -6.36 | -0.11 | +0.64 |
