# CIFAR10 Symm40 OGC Loss Weight Grid Search 全部参数

## 实验任务参数

| 参数 | 值 |
|---|---|
| dataset | seq-cifar10 |
| model | ogc-sap |
| noise_type | symm |
| noise_rate | 0.4 |
| backbone | resnet18 |
| seed | 0 |

## 训练参数

| 参数 | 值 |
|---|---:|
| n_epochs | 50 |
| batch_size | 32 |
| lr | 0.03 |
| buffer_size | 500 |
| num_workers | 4 |
| savecheck | last |

## 固定 OGC/SAP 参数

| 参数 | 值 |
|---|---:|
| ogc_low_conf_weight | 0.25 |
| ogc_buffer_penalty_coeff | 2.0 |
| sap_retain_samples | 400 |

## Grid Search 参数空间

| 参数 | 取值 |
|---|---|
| sap_scale_coff | 2000, 3000 |
| ogc_loss_weight | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 |

## 全部实验组合

| 序号 | sap_scale_coff | ogc_loss_weight | ogc_low_conf_weight | ogc_buffer_penalty_coeff | sap_retain_samples | noise_rate | noise_type | dataset | model | backbone | n_epochs | batch_size | lr | buffer_size | num_workers | seed | savecheck |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 2000 | 0.1 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 2 | 2000 | 0.2 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 3 | 2000 | 0.3 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 4 | 2000 | 0.4 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 5 | 2000 | 0.5 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 6 | 2000 | 0.6 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 7 | 2000 | 0.7 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 8 | 3000 | 0.1 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 9 | 3000 | 0.2 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 10 | 3000 | 0.3 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 11 | 3000 | 0.4 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 12 | 3000 | 0.5 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 13 | 3000 | 0.6 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |
| 14 | 3000 | 0.7 | 0.25 | 2.0 | 400 | 0.4 | symm | seq-cifar10 | ogc-sap | resnet18 | 50 | 32 | 0.03 | 500 | 4 | 0 | last |

## 命令模板

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset seq-cifar10 \
  --model ogc-sap \
  --noise_rate 0.4 \
  --noise_type symm \
  --sap_scale_coff <sap_scale_coff> \
  --ogc_loss_weight <ogc_loss_weight> \
  --ogc_low_conf_weight 0.25 \
  --ogc_buffer_penalty_coeff 2 \
  --sap_retain_samples 400 \
  --savecheck last \
  --seed 0 \
  --backbone resnet18 \
  --n_epochs 50 \
  --batch_size 32 \
  --lr 0.03 \
  --buffer_size 500 \
  --num_workers 4
```

## 串行运行顺序

| 顺序 | sap_scale_coff | ogc_loss_weight | 日志文件 |
|---:|---:|---:|---|
| 1 | 2000 | 0.1 | run_logs/cifar10_symm40_sap2000_ogc01_lcw025_pen2_seed0_serial.log |
| 2 | 2000 | 0.2 | run_logs/cifar10_symm40_sap2000_ogc02_lcw025_pen2_seed0_serial.log |
| 3 | 2000 | 0.3 | run_logs/cifar10_symm40_sap2000_ogc03_lcw025_pen2_seed0_serial.log |
| 4 | 2000 | 0.4 | run_logs/cifar10_symm40_sap2000_ogc04_lcw025_pen2_seed0_serial.log |
| 5 | 2000 | 0.5 | run_logs/cifar10_symm40_sap2000_ogc05_lcw025_pen2_seed0_serial.log |
| 6 | 2000 | 0.6 | run_logs/cifar10_symm40_sap2000_ogc06_lcw025_pen2_seed0_serial.log |
| 7 | 2000 | 0.7 | run_logs/cifar10_symm40_sap2000_ogc07_lcw025_pen2_seed0_serial.log |
| 8 | 3000 | 0.1 | run_logs/cifar10_symm40_sap3000_ogc01_lcw025_pen2_seed0_serial.log |
| 9 | 3000 | 0.2 | run_logs/cifar10_symm40_sap3000_ogc02_lcw025_pen2_seed0_serial.log |
| 10 | 3000 | 0.3 | run_logs/cifar10_symm40_sap3000_ogc03_lcw025_pen2_seed0_serial.log |
| 11 | 3000 | 0.4 | run_logs/cifar10_symm40_sap3000_ogc04_lcw025_pen2_seed0_serial.log |
| 12 | 3000 | 0.5 | run_logs/cifar10_symm40_sap3000_ogc05_lcw025_pen2_seed0_serial.log |
| 13 | 3000 | 0.6 | run_logs/cifar10_symm40_sap3000_ogc06_lcw025_pen2_seed0_serial.log |
| 14 | 3000 | 0.7 | run_logs/cifar10_symm40_sap3000_ogc07_lcw025_pen2_seed0_serial.log |
