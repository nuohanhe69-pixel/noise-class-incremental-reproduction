# CIFAR10 Symm40 ogc_low_conf_weight Grid Search 参数与结果

## 实验公共参数

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
| grid search 参数 | ogc_low_conf_weight |
| grid search 范围 | 0.05 ~ 0.50 |
| grid search 步长 | 0.05 |
| 并行方式 | parallel2 |
| RUN_TAG | 20260705_154706 |

## 实验结果

| ogc_low_conf_weight | 日志 | 完成时间 | Class-IL | Task-IL |
|---:|---|---|---:|---:|
| 0.05 | run_logs/cifar10_symm40_sap3000_ogc03_lcw005_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-05 18:36:30 | 58.79 | 88.05 |
| 0.10 | run_logs/cifar10_symm40_sap3000_ogc03_lcw010_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-05 18:36:30 | 57.85 | 87.96 |
| 0.15 | run_logs/cifar10_symm40_sap3000_ogc03_lcw015_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-05 21:28:28 | 57.26 | 87.97 |
| 0.20 | run_logs/cifar10_symm40_sap3000_ogc03_lcw020_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-05 21:28:25 | 58.56 | 87.94 |
| 0.25 | run_logs/cifar10_symm40_sap3000_ogc03_lcw025_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-06 00:21:26 | 59.77 | 88.69 |
| 0.30 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-06 00:20:40 | 61.89 | 88.61 |
| 0.35 | run_logs/cifar10_symm40_sap3000_ogc03_lcw035_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-06 03:14:50 | 61.43 | 89.02 |
| 0.40 | run_logs/cifar10_symm40_sap3000_ogc03_lcw040_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-06 03:15:28 | 61.85 | 88.87 |
| 0.45 | run_logs/cifar10_symm40_sap3000_ogc03_lcw045_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-06 06:09:13 | 61.00 | 88.18 |
| 0.50 | run_logs/cifar10_symm40_sap3000_ogc03_lcw050_pen2_seed0_lcwgrid_p2_20260705_154706.log | 2026-07-06 06:10:18 | 59.37 | 86.86 |

## Raw Accuracy

| ogc_low_conf_weight | Class-IL raw | Task-IL raw |
|---:|---|---|
| 0.05 | [57.30, 36.75, 60.30, 65.70, 73.90] | [88.50, 78.80, 86.80, 92.10, 94.05] |
| 0.10 | [46.60, 39.85, 51.90, 72.05, 78.85] | [86.55, 79.15, 87.85, 92.40, 93.85] |
| 0.15 | [61.95, 41.90, 50.75, 59.25, 72.45] | [89.75, 79.55, 84.85, 92.15, 93.55] |
| 0.20 | [60.15, 45.15, 54.60, 65.85, 67.05] | [87.00, 79.60, 87.15, 91.75, 94.20] |
| 0.25 | [62.75, 44.35, 61.50, 62.05, 68.20] | [90.85, 78.30, 88.55, 90.85, 94.90] |
| 0.30 | [68.15, 48.40, 53.70, 69.65, 69.55] | [88.40, 79.85, 87.45, 93.25, 94.10] |
| 0.35 | [64.10, 51.70, 46.65, 73.85, 70.85] | [89.10, 79.10, 88.70, 93.50, 94.70] |
| 0.40 | [64.55, 42.20, 60.60, 68.95, 72.95] | [88.75, 80.55, 86.60, 94.40, 94.05] |
| 0.45 | [63.60, 53.20, 51.75, 67.45, 69.00] | [85.55, 82.75, 84.85, 92.85, 94.90] |
| 0.50 | [55.50, 49.65, 56.70, 61.60, 73.40] | [86.00, 78.75, 85.70, 91.05, 92.80] |
