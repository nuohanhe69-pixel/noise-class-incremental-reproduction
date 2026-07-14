# CIFAR10 Symm40 sap=2000 ogc=0.4 同参数实验对比

## 项目文件路径

| 项目 | 路径 |
|---|---|
| 当前本地项目 | /Users/hunk/Documents/deep learning/噪声类增量项目复现 |
| 当前本地代码目录 | /Users/hunk/Documents/deep learning/噪声类增量项目复现/mammoth_code |
| 远程实验目录 | /home/hnh/mammoth_code_cifar10 |

## 当前实验

| 项目 | 值 |
|---|---|
| 当前实验日志 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log |
| sap_scale_coff | 2000 |
| ogc_loss_weight | 0.40 |
| ogc_low_conf_weight | 0.25 |
| ogc_buffer_penalty_coeff | 2.0 |
| sap_retain_samples | 400 |
| 最终 Class-IL | 57.56 |
| 最终 Task-IL | 86.39 |
| 最终结果时间 | 2026-07-02 21:11:58 |

## 同参数实验清单与时间

| 序号 | 实验来源 | 运行名 / 日志 | 时间记录 | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | 原文对齐组 | cifar10_symm40_sap2000_seed0 | 记录文件时间：2026-06-27 23:29:13 | 2000 | 0.40 | 0.25 | 2.0 | 400 |
| 2 | sap_scale_coff grid search | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 记录文件时间：2026-06-30 16:19:22 | 2000 | 0.40 | 0.25 | 2.0 | 400 |
| 3 | ogc_loss_weight grid search 并行运行 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 记录文件时间：2026-07-02 10:58:36 | 2000 | 0.40 | 0.25 | 2.0 | 400 |
| 4 | ogc_loss_weight grid search 串行重跑 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log | 最终结果时间：2026-07-02 21:11:58 | 2000 | 0.40 | 0.25 | 2.0 | 400 |

## 实验参数对比

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

## 结果指标对比

| 实验来源 | 运行名 / 日志 | Class-IL | Task-IL | 相对当前串行 ΔClass-IL | 相对当前串行 ΔTask-IL |
|---|---|---:|---:|---:|---:|
| 原文对齐组 | cifar10_symm40_sap2000_seed0 | 58.21 | 87.74 | +0.65 | +1.35 |
| sap_scale_coff grid search | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 60.56 | 87.44 | +3.00 | +1.05 |
| ogc_loss_weight grid search 并行运行 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_grid.log | 59.68 | 85.78 | +2.12 | -0.61 |
| ogc_loss_weight grid search 串行重跑 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log | 57.56 | 86.39 | 0.00 | 0.00 |

## 对标实验结果差值

| 对标实验 | 对标 Class-IL | 当前串行 Class-IL | 当前-对标 ΔClass-IL | 对标 Task-IL | 当前串行 Task-IL | 当前-对标 ΔTask-IL |
|---|---:|---:|---:|---:|---:|---:|
| sap_scale_coff grid search 中的 sap=2000 | 60.56 | 57.56 | -3.00 | 87.44 | 86.39 | -1.05 |

## 历史实验 Raw Accuracy

| 实验来源 | Class-IL raw accuracies | Task-IL raw accuracies |
|---|---|---|
| 原文对齐组 | [59.40, 49.10, 52.35, 66.55, 63.65] | [85.90, 80.15, 87.30, 92.50, 92.85] |
| sap_scale_coff grid search | [66.45, 51.95, 52.85, 63.55, 68.00] | [87.40, 79.85, 85.15, 90.20, 94.60] |
| ogc_loss_weight grid search 并行运行 | [53.85, 50.00, 52.45, 73.30, 68.80] | [79.85, 77.75, 85.30, 93.15, 92.85] |

## 当前串行重跑逐任务累计结果

| task 数 | 时间 | Class-IL | Task-IL |
|---:|---|---:|---:|
| 1 | 2026-07-02 19:16:52 | 94.95 | 94.95 |
| 2 | 2026-07-02 19:44:58 | 77.28 | 86.65 |
| 3 | 2026-07-02 20:13:30 | 68.23 | 87.12 |
| 4 | 2026-07-02 20:42:24 | 61.42 | 86.90 |
| 5 | 2026-07-02 21:11:58 | 57.56 | 86.39 |

## 数据来源文件

| 内容 | 来源 |
|---|---|
| 原文对齐组结果 | cifar10_symm40_symm60_tuning_summary.md |
| sap_scale_coff grid search 结果 | cifar10_symm40_sap_grid_search_summary.md |
| ogc_loss_weight grid search 并行结果 | cifar10_symm40_ogc_grid_compare.md |
| 当前串行重跑结果 | 用户截图：grep -n "Accuracy for" run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log |
