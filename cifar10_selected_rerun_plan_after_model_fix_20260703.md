# CIFAR10 模型修正后选择性重跑实验

## 选择依据

| 来源文件 | 使用内容 |
|---|---|
| cifar10_all_experiments_parameters_results_20260703.md | 历史实验参数、Class-IL、Task-IL、raw accuracy |
| readme_latest.md | CIFAR10 推荐复现参数 |

## CIFAR10 大实验范围

| 大实验 | noise_type | noise_rate | 论文 Class-IL |
|---|---|---:|---|
| CIFAR10 Symmetric 20% | symm | 0.2 | 67.64 ± 0.52 |
| CIFAR10 Symmetric 40% | symm | 0.4 | 64.46 ± 0.40 |
| CIFAR10 Symmetric 60% | symm | 0.6 | 47.66 ± 0.59 |
| CIFAR10 Asymmetric 20% | asym | 0.2 | 未记录 |
| CIFAR10 Asymmetric 40% | asym | 0.4 | 57.60 ± 0.70 |

## 公共参数

| 参数 | 值 |
|---|---|
| dataset | seq-cifar10 |
| model | ogc-sap |
| backbone | resnet18 |
| n_epochs | 50 |
| batch_size | 32 |
| lr | 0.03 |
| buffer_size | 500 |
| num_workers | 4 |
| savecheck | last |
| seed | 0 |

## README 推荐组与历史较优组差异

| 大实验 | README 推荐参数 | 历史较优参数 | 差异参数 |
|---|---|---|---|
| Symm20 | sap=1000, ogc=0.30, lcw=0.50, pen=1.5, retain=400 | sap=1000, ogc=0.30, lcw=0.30, pen=2.0, retain=400 | ogc_low_conf_weight, ogc_buffer_penalty_coeff |
| Symm40 | sap=2000, ogc=0.40, lcw=0.30, pen=2.0, retain=300 | sap=3000, ogc=0.40, lcw=0.25, pen=2.0, retain=400 | sap_scale_coff, ogc_low_conf_weight, sap_retain_samples |
| Symm60 | sap=3000, ogc=0.50, lcw=0.20, pen=2.5, retain=200 | sap=5000, ogc=0.55, lcw=0.15, pen=2.5, retain=400 | sap_scale_coff, ogc_loss_weight, ogc_low_conf_weight, sap_retain_samples |
| Asym20 | sap=1000, ogc=0.30, lcw=0.50, pen=1.5, retain=400 | 无历史结果 | 无法对比 |
| Asym40 | sap=700, ogc=0.40, lcw=0.30, pen=2.0, retain=400 | sap=1000, ogc=0.40, lcw=0.30, pen=2.0, retain=400 | sap_scale_coff |

## 第一轮优先重跑

| 优先级 | 大实验 | 小实验来源 | run_name 建议 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | 历史 Class-IL | 历史 Task-IL | 选择原因 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0 | Symm20 | 历史最优 | fix_cifar10_symm20_sap1000_ogc03_lcw03_pen2_ret400_seed0 | symm | 0.2 | 1000 | 0.30 | 0.30 | 2.0 | 400 | 66.81 | 91.20 | Symm20 历史 Class-IL 最高 |
| P0 | Symm40 | 历史前置最优 | fix_cifar10_symm40_sap3000_ogc04_lcw025_pen2_ret400_seed0 | symm | 0.4 | 3000 | 0.40 | 0.25 | 2.0 | 400 | 60.74 | 88.28 | Symm40 前置实验 Class-IL 最高 |
| P0 | Symm40 | SAP grid 最优 | fix_cifar10_symm40_sap2000_ogc04_lcw025_pen2_ret400_seed0 | symm | 0.4 | 2000 | 0.40 | 0.25 | 2.0 | 400 | 60.56 | 87.44 | SAP grid 中 Class-IL 最高 |
| P0 | Symm60 | 历史最优 | fix_cifar10_symm60_sap5000_ogc055_lcw015_pen25_ret400_seed0 | symm | 0.6 | 5000 | 0.55 | 0.15 | 2.5 | 400 | 45.27 | 80.93 | Symm60 历史 Class-IL 最高 |
| P0 | Asym40 | 历史最优 | fix_cifar10_asym40_sap1000_ogc04_lcw03_pen2_ret400_seed0 | asym | 0.4 | 1000 | 0.40 | 0.30 | 2.0 | 400 | 56.96 | 91.27 | Asym40 历史 Class-IL 最高且进入论文区间 |
| P0 | Asym20 | README 推荐 | fix_cifar10_asym20_sap1000_ogc03_lcw05_pen15_ret400_seed0 | asym | 0.2 | 1000 | 0.30 | 0.50 | 1.5 | 400 | 未记录 | 未记录 | README 有推荐参数，历史文件无正式结果 |

## 第二轮候选重跑

| 优先级 | 大实验 | 小实验来源 | run_name 建议 | noise_type | noise_rate | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | 历史 Class-IL | 历史 Task-IL | 选择原因 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P1 | Symm20 | 历史次优 | fix_cifar10_symm20_sap3000_ogc04_lcw02_pen3_ret400_seed0 | symm | 0.2 | 3000 | 0.40 | 0.20 | 3.0 | 400 | 65.13 | 90.06 | Symm20 历史第二梯队，OGC 更强、SAP 更强 |
| P1 | Symm20 | README 推荐 | fix_cifar10_symm20_sap1000_ogc03_lcw05_pen15_ret400_seed0 | symm | 0.2 | 1000 | 0.30 | 0.50 | 1.5 | 400 | 未记录 | 未记录 | README 推荐，与历史最优只差 lcw 和 pen |
| P1 | Symm40 | OGC 串行较优 | fix_cifar10_symm40_sap2000_ogc01_lcw025_pen2_ret400_seed0 | symm | 0.4 | 2000 | 0.10 | 0.25 | 2.0 | 400 | 60.37 | 88.90 | sap=2000 串行重跑 Class-IL 最高、Task-IL 最高 |
| P1 | Symm40 | OGC grid 较优 | fix_cifar10_symm40_sap3000_ogc02_lcw025_pen2_ret400_seed0 | symm | 0.4 | 3000 | 0.20 | 0.25 | 2.0 | 400 | 59.64 | 88.48 | sap=3000 并行结果中 Class-IL 和 Task-IL 最高 |
| P1 | Symm40 | README 推荐 | fix_cifar10_symm40_sap2000_ogc04_lcw03_pen2_ret300_seed0 | symm | 0.4 | 2000 | 0.40 | 0.30 | 2.0 | 300 | 未记录 | 未记录 | README 推荐，与历史较优参数不同 |
| P1 | Symm60 | 原文对齐组 | fix_cifar10_symm60_sap3000_ogc05_lcw02_pen2_ret400_seed0 | symm | 0.6 | 3000 | 0.50 | 0.20 | 2.0 | 400 | 45.05 | 79.82 | Class-IL 接近 Symm60 历史最优 |
| P1 | Symm60 | README 推荐 | fix_cifar10_symm60_sap3000_ogc05_lcw02_pen25_ret200_seed0 | symm | 0.6 | 3000 | 0.50 | 0.20 | 2.5 | 200 | 未记录 | 未记录 | README 推荐，与历史较优参数不同 |
| P1 | Asym40 | 历史次优 | fix_cifar10_asym40_sap1000_ogc045_lcw025_pen25_ret400_seed0 | asym | 0.4 | 1000 | 0.45 | 0.25 | 2.5 | 400 | 56.75 | 91.87 | Class-IL 接近最优，Task-IL 最高 |
| P1 | Asym40 | README 推荐 | fix_cifar10_asym40_sap700_ogc04_lcw03_pen2_ret400_seed0 | asym | 0.4 | 700 | 0.40 | 0.30 | 2.0 | 400 | 52.16 | 89.69 | README 推荐，但历史结果低于 sap=1000 |

## 暂不优先重跑

| 大实验 | 暂不优先组合 | 历史 Class-IL | 暂不优先原因 |
|---|---|---:|---|
| Symm20 | num_workers=0 的早期本机实验 | 60.47 ~ 63.45 | 远程正式设置使用 num_workers=4，且结果不优 |
| Symm20 | sap=5000, ogc=0.40, lcw=0.20, pen=3.0, retain=400 | 65.13 | 与 sap=3000 同结果，第一轮先保留 sap=3000 |
| Symm40 | sap=1000, 1500, 2500, 3500, 4000, 4500, 5000 的 SAP grid 其他组合 | 54.23 ~ 59.78 | Class-IL 低于 sap=2000/3000 主候选 |
| Symm40 | sap=2000 的 ogc=0.2/0.3/0.5/0.6/0.7 并行结果 | 56.46 ~ 58.58 | 低于 sap=2000/ogc=0.4 与串行 ogc=0.1 |
| Symm40 | sap=2000 的 ogc=0.2/0.3/0.4/0.5/0.6/0.7 串行结果 | 55.82 ~ 59.52 | 低于串行 ogc=0.1 |
| Symm60 | sap=5000, ogc=0.50, lcw=0.20, pen=2.0, retain=400 | 44.54 | 低于 Symm60 P0/P1 候选 |
| Symm60 | sap=3000, ogc=0.55, lcw=0.15, pen=3.0, retain=400 | 44.38 | Class-IL 低于 P0/P1，Task-IL 可作为后续备选 |
| Asym40 | sap=700, ogc=0.40, lcw=0.30, pen=2.0, retain=400 | 52.16 | 作为 README 对照可跑，但不作为第一轮核心 |

## 第一轮命令模板

```bash
COMMON_ARGS="--backbone resnet18 --n_epochs 50 --batch_size 32 --lr 0.03 --buffer_size 500 --num_workers 4 --savecheck last --seed 0"
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.2 --noise_type symm --sap_scale_coff 1000 --ogc_loss_weight 0.30 --ogc_low_conf_weight 0.30 --ogc_buffer_penalty_coeff 2.0 --sap_retain_samples 400 ${COMMON_ARGS}
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.4 --noise_type symm --sap_scale_coff 3000 --ogc_loss_weight 0.40 --ogc_low_conf_weight 0.25 --ogc_buffer_penalty_coeff 2.0 --sap_retain_samples 400 ${COMMON_ARGS}
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.4 --noise_type symm --sap_scale_coff 2000 --ogc_loss_weight 0.40 --ogc_low_conf_weight 0.25 --ogc_buffer_penalty_coeff 2.0 --sap_retain_samples 400 ${COMMON_ARGS}
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.6 --noise_type symm --sap_scale_coff 5000 --ogc_loss_weight 0.55 --ogc_low_conf_weight 0.15 --ogc_buffer_penalty_coeff 2.5 --sap_retain_samples 400 ${COMMON_ARGS}
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.4 --noise_type asym --sap_scale_coff 1000 --ogc_loss_weight 0.40 --ogc_low_conf_weight 0.30 --ogc_buffer_penalty_coeff 2.0 --sap_retain_samples 400 ${COMMON_ARGS}
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.2 --noise_type asym --sap_scale_coff 1000 --ogc_loss_weight 0.30 --ogc_low_conf_weight 0.50 --ogc_buffer_penalty_coeff 1.5 --sap_retain_samples 400 ${COMMON_ARGS}
```
