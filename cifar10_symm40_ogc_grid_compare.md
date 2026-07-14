# CIFAR10 Symm40 OGC Loss Weight Grid Search 结果对比

## 公共超参数

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

## 对标实验参数与结果

| 对标组 | 来源 | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | Class-IL | Task-IL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sap=2000 对标 | sap_scale_coff grid search 中的 sap=2000 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 60.56 | 87.44 |
| sap=3000 对标 | sap_scale_coff grid search 中的 sap=3000 | 3000 | 0.40 | 0.25 | 2.0 | 400 | 60.54 | 88.09 |

## 对标实验 Raw Accuracy

| 对标组 | Class-IL raw accuracies | Task-IL raw accuracies |
|---|---|---|
| sap=2000 对标 | [66.45, 51.95, 52.85, 63.55, 68.00] | [87.40, 79.85, 85.15, 90.20, 94.60] |
| sap=3000 对标 | [60.25, 50.75, 57.10, 62.95, 71.65] | [89.25, 80.20, 86.25, 93.25, 91.50] |

## sap_scale_coff=2000 Grid Search 结果

| sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | Class-IL | Task-IL | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2000 | 0.1 | 0.25 | 2.0 | 400 | 57.60 | 86.57 | DONE |
| 2000 | 0.2 | 0.25 | 2.0 | 400 | 58.41 | 86.15 | DONE |
| 2000 | 0.3 | 0.25 | 2.0 | 400 | 57.89 | 87.93 | DONE |
| 2000 | 0.4 | 0.25 | 2.0 | 400 | 59.68 | 85.78 | DONE |
| 2000 | 0.5 | 0.25 | 2.0 | 400 | 56.46 | 87.20 | DONE |
| 2000 | 0.6 | 0.25 | 2.0 | 400 | 56.70 | 87.90 | DONE |
| 2000 | 0.7 | 0.25 | 2.0 | 400 | 58.58 | 86.68 | DONE |

## sap_scale_coff=2000 Raw Accuracy

| ogc_loss_weight | Class-IL raw accuracies | Task-IL raw accuracies |
|---:|---|---|
| 0.1 | [58.35, 44.80, 53.90, 71.95, 59.00] | [83.60, 80.45, 81.60, 94.20, 93.00] |
| 0.2 | [63.60, 46.40, 54.55, 63.10, 64.40] | [87.15, 77.40, 84.20, 91.30, 90.70] |
| 0.3 | [49.80, 48.20, 51.45, 68.80, 71.20] | [85.85, 81.30, 86.25, 92.70, 93.55] |
| 0.4 | [53.85, 50.00, 52.45, 73.30, 68.80] | [79.85, 77.75, 85.30, 93.15, 92.85] |
| 0.5 | [65.65, 44.40, 59.65, 62.45, 50.15] | [84.40, 78.20, 87.85, 92.25, 93.30] |
| 0.6 | [61.65, 45.80, 54.85, 65.00, 56.20] | [88.10, 80.35, 85.25, 93.15, 92.65] |
| 0.7 | [60.60, 56.65, 41.30, 67.80, 66.55] | [86.80, 78.75, 83.10, 90.65, 94.10] |

## sap_scale_coff=2000 相对对标组超参数对比

| ogc_loss_weight | Δogc_loss_weight | sap_scale_coff | Δsap_scale_coff | ogc_low_conf_weight | Δogc_low_conf_weight | ogc_buffer_penalty_coeff | Δogc_buffer_penalty_coeff | sap_retain_samples | Δsap_retain_samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.30 | 2000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.2 | -0.20 | 2000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.3 | -0.10 | 2000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.4 | 0.00 | 2000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.5 | +0.10 | 2000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.6 | +0.20 | 2000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.7 | +0.30 | 2000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |

## sap_scale_coff=2000 相对对标组结果指标对比

| ogc_loss_weight | Class-IL | ΔClass-IL | Task-IL | ΔTask-IL |
|---:|---:|---:|---:|---:|
| 0.1 | 57.60 | -2.96 | 86.57 | -0.87 |
| 0.2 | 58.41 | -2.15 | 86.15 | -1.29 |
| 0.3 | 57.89 | -2.67 | 87.93 | +0.49 |
| 0.4 | 59.68 | -0.88 | 85.78 | -1.66 |
| 0.5 | 56.46 | -4.10 | 87.20 | -0.24 |
| 0.6 | 56.70 | -3.86 | 87.90 | +0.46 |
| 0.7 | 58.58 | -1.98 | 86.68 | -0.76 |

## sap_scale_coff=3000 Grid Search 结果

| sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | Class-IL | Task-IL | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 3000 | 0.1 | 0.25 | 2.0 | 400 | 55.76 | 84.35 | DONE |
| 3000 | 0.2 | 0.25 | 2.0 | 400 | 59.64 | 88.48 | DONE |
| 3000 | 0.3 | 0.25 | 2.0 | 400 | 55.82 | 87.04 | DONE |

## sap_scale_coff=3000 Raw Accuracy

| ogc_loss_weight | Class-IL raw accuracies | Task-IL raw accuracies |
|---:|---|---|
| 0.1 | [62.25, 57.15, 39.20, 60.75, 59.45] | [88.25, 80.60, 80.15, 87.90, 84.85] |
| 0.2 | [62.30, 56.35, 45.75, 67.65, 66.15] | [86.30, 82.70, 85.00, 93.45, 94.95] |
| 0.3 | [61.85, 52.55, 51.55, 67.75, 45.40] | [85.50, 80.35, 87.30, 90.00, 92.05] |

## sap_scale_coff=3000 相对对标组超参数对比

| ogc_loss_weight | Δogc_loss_weight | sap_scale_coff | Δsap_scale_coff | ogc_low_conf_weight | Δogc_low_conf_weight | ogc_buffer_penalty_coeff | Δogc_buffer_penalty_coeff | sap_retain_samples | Δsap_retain_samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.30 | 3000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.2 | -0.20 | 3000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |
| 0.3 | -0.10 | 3000 | 0 | 0.25 | 0.00 | 2.0 | 0.0 | 400 | 0 |

## sap_scale_coff=3000 相对对标组结果指标对比

| ogc_loss_weight | Class-IL | ΔClass-IL | Task-IL | ΔTask-IL |
|---:|---:|---:|---:|---:|
| 0.1 | 55.76 | -4.78 | 84.35 | -3.74 |
| 0.2 | 59.64 | -0.90 | 88.48 | +0.39 |
| 0.3 | 55.82 | -4.72 | 87.04 | -1.05 |
