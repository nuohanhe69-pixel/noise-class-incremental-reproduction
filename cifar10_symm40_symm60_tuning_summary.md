# CIFAR10 Symm40 / Symm60 四组实验汇总

统计日期：2026-06-27

## 1. 论文目标指标

| 实验 | 论文 Table I Ours 指标 | 论文波动区间 |
|---|---:|---:|
| CIFAR10 Symmetric 40% | 64.46 ± 0.40 | 64.06 ~ 64.86 |
| CIFAR10 Symmetric 60% | 47.66 ± 0.59 | 47.07 ~ 48.25 |

## 2. 公共实验参数

四组实验均使用以下公共参数：

```bash
--dataset seq-cifar10
--model ogc-sap
--noise_type symm
--backbone resnet18
--n_epochs 50
--batch_size 32
--lr 0.03
--buffer_size 500
--num_workers 4
--sap_retain_samples 400
--savecheck last
--seed 0
```

## 3. 四组实验参数与结果总表

| 组别 | 实验 | 运行名 | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | Class-IL | Task-IL | 与论文均值差距 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 第一组 A | Symm40-A | cifar10_symm40_sap3000_ogc04_lcw025_pen2 | 0.4 | 3000 | 0.40 | 0.25 | 2.0 | 60.74 | 88.28 | -3.72 |
| 第一组 A | Symm60-A | cifar10_symm60_sap5000_ogc05_lcw02_pen2 | 0.6 | 5000 | 0.50 | 0.20 | 2.0 | 44.54 | 80.32 | -3.12 |
| 第二组 B | Symm40-B | cifar10_symm40_sap5000_ogc04_lcw025_pen2 | 0.4 | 5000 | 0.40 | 0.25 | 2.0 | 57.99 | 87.25 | -6.47 |
| 第二组 B | Symm60-B | cifar10_symm60_sap5000_ogc055_lcw015_pen25 | 0.6 | 5000 | 0.55 | 0.15 | 2.5 | 45.27 | 80.93 | -2.39 |
| 第三组 C | Symm40-C | cifar10_symm40_sap3000_ogc045_lcw02_pen25 | 0.4 | 3000 | 0.45 | 0.20 | 2.5 | 56.63 | 86.13 | -7.83 |
| 第三组 C | Symm60-C | cifar10_symm60_sap3000_ogc055_lcw015_pen3 | 0.6 | 3000 | 0.55 | 0.15 | 3.0 | 44.38 | 81.13 | -3.28 |
| 第四组 D | Symm40-D | cifar10_symm40_sap2000_seed0 | 0.4 | 2000 | 0.40 | 0.25 | 2.0 | 58.21 | 87.74 | -6.25 |
| 第四组 D | Symm60-D | cifar10_symm60_sap3000_seed0 | 0.6 | 3000 | 0.50 | 0.20 | 2.0 | 45.05 | 79.82 | -2.61 |

## 4. 每组详细结果

### 第一组 A

#### Symm40-A

参数：

```bash
--noise_rate 0.4
--noise_type symm
--sap_scale_coff 3000
--ogc_loss_weight 0.4
--ogc_low_conf_weight 0.25
--ogc_buffer_penalty_coeff 2
```

结果：

```text
Class-IL: 60.74%
Task-IL: 88.28%

Class-IL raw:
[64.05, 55.90, 47.25, 71.80, 64.70]

Task-IL raw:
[89.40, 81.65, 85.00, 91.35, 94.00]
```

与论文 Symm40 目标 `64.46 ± 0.40` 对比：

```text
60.74 - 64.46 = -3.72
```

#### Symm60-A

参数：

```bash
--noise_rate 0.6
--noise_type symm
--sap_scale_coff 5000
--ogc_loss_weight 0.5
--ogc_low_conf_weight 0.2
--ogc_buffer_penalty_coeff 2
```

结果：

```text
Class-IL: 44.54%
Task-IL: 80.32%

Class-IL raw:
[48.05, 35.25, 30.95, 53.45, 55.00]

Task-IL raw:
[85.65, 71.00, 76.95, 85.70, 82.30]
```

与论文 Symm60 目标 `47.66 ± 0.59` 对比：

```text
44.54 - 47.66 = -3.12
```

### 第二组 B

#### Symm40-B

参数：

```bash
--noise_rate 0.4
--noise_type symm
--sap_scale_coff 5000
--ogc_loss_weight 0.4
--ogc_low_conf_weight 0.25
--ogc_buffer_penalty_coeff 2
```

结果：

```text
Class-IL: 57.99%
Task-IL: 87.25%

Class-IL raw:
[65.35, 56.50, 43.65, 59.95, 64.50]

Task-IL raw:
[87.85, 79.35, 82.85, 91.60, 94.60]
```

与论文 Symm40 目标 `64.46 ± 0.40` 对比：

```text
57.99 - 64.46 = -6.47
```

#### Symm60-B

参数：

```bash
--noise_rate 0.6
--noise_type symm
--sap_scale_coff 5000
--ogc_loss_weight 0.55
--ogc_low_conf_weight 0.15
--ogc_buffer_penalty_coeff 2.5
```

结果：

```text
Class-IL: 45.27%
Task-IL: 80.93%

Class-IL raw:
[43.00, 26.75, 39.70, 50.10, 66.80]

Task-IL raw:
[82.05, 73.95, 76.15, 86.10, 86.40]
```

与论文 Symm60 目标 `47.66 ± 0.59` 对比：

```text
45.27 - 47.66 = -2.39
```

### 第三组 C

#### Symm40-C

参数：

```bash
--noise_rate 0.4
--noise_type symm
--sap_scale_coff 3000
--ogc_loss_weight 0.45
--ogc_low_conf_weight 0.2
--ogc_buffer_penalty_coeff 2.5
```

结果：

```text
Class-IL: 56.63%
Task-IL: 86.13%

Class-IL raw:
[57.85, 37.00, 58.80, 71.55, 57.95]

Task-IL raw:
[88.00, 77.45, 85.95, 92.40, 86.85]
```

与论文 Symm40 目标 `64.46 ± 0.40` 对比：

```text
56.63 - 64.46 = -7.83
```

#### Symm60-C

参数：

```bash
--noise_rate 0.6
--noise_type symm
--sap_scale_coff 3000
--ogc_loss_weight 0.55
--ogc_low_conf_weight 0.15
--ogc_buffer_penalty_coeff 3
```

结果：

```text
Class-IL: 44.38%
Task-IL: 81.13%

Class-IL raw:
[50.35, 30.10, 36.50, 56.90, 48.05]

Task-IL raw:
[82.20, 72.60, 79.35, 85.05, 86.45]
```

与论文 Symm60 目标 `47.66 ± 0.59` 对比：

```text
44.38 - 47.66 = -3.28
```

### 第四组 D

#### Symm40-D

参数：

```bash
--noise_rate 0.4
--noise_type symm
--sap_scale_coff 2000
--ogc_loss_weight 0.4
--ogc_low_conf_weight 0.25
--ogc_buffer_penalty_coeff 2
```

结果：

```text
Class-IL: 58.21%
Task-IL: 87.74%

Class-IL raw:
[59.40, 49.10, 52.35, 66.55, 63.65]

Task-IL raw:
[85.90, 80.15, 87.30, 92.50, 92.85]
```

与论文 Symm40 目标 `64.46 ± 0.40` 对比：

```text
58.21 - 64.46 = -6.25
```

#### Symm60-D

参数：

```bash
--noise_rate 0.6
--noise_type symm
--sap_scale_coff 3000
--ogc_loss_weight 0.5
--ogc_low_conf_weight 0.2
--ogc_buffer_penalty_coeff 2
```

结果：

```text
Class-IL: 45.05%
Task-IL: 79.82%

Class-IL raw:
[55.15, 26.75, 35.00, 55.70, 52.65]

Task-IL raw:
[85.10, 64.35, 74.20, 88.05, 87.40]
```

与论文 Symm60 目标 `47.66 ± 0.59` 对比：

```text
45.05 - 47.66 = -2.61
```

## 5. 当前最优方案

| 实验 | 当前最优方案 | Class-IL | Task-IL | 与论文均值差距 |
|---|---|---:|---:|---:|
| Symm40 | 第一组 A：sap=3000, ogc=0.4, lcw=0.25, pen=2 | 60.74 | 88.28 | -3.72 |
| Symm60 | 第二组 B：sap=5000, ogc=0.55, lcw=0.15, pen=2.5 | 45.27 | 80.93 | -2.39 |
