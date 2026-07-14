# CIFAR10 Symm40 ogc_buffer_penalty_coeff Grid Search 参数与结果

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
| ogc_low_conf_weight | 0.30 |
| sap_retain_samples | 400 |
| grid search 参数 | ogc_buffer_penalty_coeff |
| grid search 范围 | 0.5 ~ 5.0 |
| grid search 步长 | 0.5 |
| 并行方式 | parallel2 |
| RUN_TAG | 20260706_161508 |

## 实验结果

| ogc_buffer_penalty_coeff | 日志 | 完成时间 | Class-IL | Task-IL |
|---:|---|---|---:|---:|
| 0.5 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen05_seed0_pengrid_p2_20260706_161508.log | 2026-07-06 19:55:09 | 33.03 | 83.22 |
| 1.0 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen10_seed0_pengrid_p2_20260706_161508.log | 2026-07-06 19:55:09 | 53.48 | 86.71 |
| 1.5 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen15_seed0_pengrid_p2_20260706_161508.log | 2026-07-06 23:18:07 | 57.49 | 88.35 |
| 2.0 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen20_seed0_pengrid_p2_20260706_161508.log | 2026-07-06 23:18:05 | 60.36 | 87.82 |
| 2.5 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen25_seed0_pengrid_p2_20260706_161508.log | 2026-07-07 02:18:59 | 61.65 | 88.39 |
| 3.0 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen30_seed0_pengrid_p2_20260706_161508.log | 2026-07-07 02:19:46 | 59.46 | 88.26 |
| 3.5 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen35_seed0_pengrid_p2_20260706_161508.log | 2026-07-07 05:15:27 | 61.17 | 88.39 |
| 4.0 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen40_seed0_pengrid_p2_20260706_161508.log | 2026-07-07 05:16:10 | 62.93 | 90.07 |
| 4.5 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen45_seed0_pengrid_p2_20260706_161508.log | 2026-07-07 08:12:15 | 57.83 | 87.22 |
| 5.0 | run_logs/cifar10_symm40_sap3000_ogc03_lcw030_pen50_seed0_pengrid_p2_20260706_161508.log | 2026-07-07 08:13:36 | 61.52 | 89.27 |

## Raw Accuracy

| ogc_buffer_penalty_coeff | Class-IL raw | Task-IL raw |
|---:|---|---|
| 0.5 | [38.15, 25.20, 31.25, 32.70, 37.85] | [85.85, 73.75, 77.50, 86.50, 92.50] |
| 1.0 | [56.80, 35.70, 44.35, 69.85, 60.70] | [88.90, 75.85, 81.70, 92.90, 94.20] |
| 1.5 | [64.75, 42.30, 48.55, 71.30, 60.55] | [90.10, 78.80, 85.05, 92.20, 95.60] |
| 2.0 | [63.95, 47.15, 48.85, 67.15, 74.70] | [87.95, 80.35, 83.75, 93.10, 93.95] |
| 2.5 | [64.80, 48.10, 50.90, 70.50, 73.95] | [88.85, 78.70, 87.25, 92.00, 95.15] |
| 3.0 | [62.80, 57.25, 52.40, 60.55, 64.30] | [88.00, 81.65, 84.20, 93.55, 93.90] |
| 3.5 | [64.55, 54.70, 48.95, 71.20, 66.45] | [88.15, 80.10, 86.65, 92.55, 94.50] |
| 4.0 | [68.90, 48.95, 63.85, 63.50, 69.45] | [93.70, 80.80, 88.40, 92.85, 94.60] |
| 4.5 | [67.35, 44.30, 47.00, 65.35, 65.15] | [90.55, 79.65, 84.10, 91.75, 90.05] |
| 5.0 | [60.90, 46.50, 47.45, 71.20, 81.55] | [90.25, 79.70, 87.25, 93.90, 95.25] |
