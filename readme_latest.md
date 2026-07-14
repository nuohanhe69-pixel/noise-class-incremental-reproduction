# Latest Reproduction Commands

本文档整理当前仓库中噪声类增量学习实验的推荐复现命令。命令格式保持为先定义 `COMMON_ARGS`，再按数据集、噪声类型和噪声比例分别运行。
## 环境配置参考README.md, 建议使用conda环境,由于此代码是在baseline基础上改进的，所以配置完以后需要执行后续命令，看缺少哪些变量在进行配置

## 基础说明

主要模型：

```bash
# AER/ABS baseline
--model er-ace-aer-abs
# AER/ABS + SAP
--model aer-sap
# AER/ABS + OGC
--model ogc-sap --enable_sap 0
# AER/ABS + OGC + SAP
--model ogc-sap --enable_sap 1
```

噪声类型：

```bash
--noise_type symm   # symmetric noise
--noise_type asym   # asymmetric noise
```

建议正式实验都加：

```bash
--savecheck last     注：savecheck会保留权重用于复现，可不加，加了会保存于checkpoints文件夹下，并可以通过1.py转换为可用于复现的权重文件
--seed 0 或者--seed 152   注：seed为随机种子用于复现实验，建议实验时固定为0或152，其他值会导致实验结果不同
```

如果服务器出现 multiprocessing 临时目录或 DataLoader 卡住问题，可以额外加：

```bash
--num_workers 0
```

## 基于权重文件的复现
用于使用已保存的权重文件  复现cifar100在symm噪声模式60噪声比例的实验
```bash
python main.py   --loadcheck checkpoints/cifar100_symm_60_eval_best.pt   --inference_only 1   --start_from 9   --stop_after 10
```
```bash
python main.py   --loadcheck checkpoints/cifar100_symm_60_eval.pt   --inference_only 1   --start_from 9   --stop_after 10
```


## CIFAR100

CIFAR100 推荐 `buffer_size=2000`。

```bash
COMMON_ARGS="--backbone resnet18 --n_epochs 50 --batch_size 32 --lr 0.03 --buffer_size 2000 --num_workers 4"
```

### CIFAR100 Symmetric 20%

```bash
python main.py --dataset seq-cifar100 --model ogc-sap --noise_rate 0.2 --noise_type symm \
  --sap_scale_coff 1000 \
  --ogc_loss_weight 0.2 \
  --ogc_low_conf_weight 0.5 \
  --ogc_buffer_penalty_coeff 1.5 \
  --sap_retain_samples 1500 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR100 Symmetric 40%

```bash
python main.py --dataset seq-cifar100 --model ogc-sap --noise_rate 0.4 --noise_type symm \
  --sap_scale_coff 3000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.4 \
  --ogc_buffer_penalty_coeff 1.8 \
  --sap_retain_samples 1500 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR100 Symmetric 60%

```bash
python main.py --dataset seq-cifar100 --model ogc-sap --noise_rate 0.6 --noise_type symm \
  --sap_scale_coff 5000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 1500 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR100 Asymmetric 20%

```bash
python main.py --dataset seq-cifar100 --model ogc-sap --noise_rate 0.2 --noise_type asym \
  --sap_scale_coff 1000 \
  --ogc_loss_weight 0.2 \
  --ogc_low_conf_weight 0.5 \
  --ogc_buffer_penalty_coeff 1.5 \
  --sap_retain_samples 1500 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR100 Asymmetric 40%

```bash
python main.py --dataset seq-cifar100 --model ogc-sap --noise_rate 0.4 --noise_type asym \
  --sap_scale_coff 3000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.4 \
  --ogc_buffer_penalty_coeff 1.8 \
  --sap_retain_samples 1500 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

## CIFAR10

CIFAR10 推荐 `buffer_size=500`。

```bash
COMMON_ARGS="--backbone resnet18 --n_epochs 50 --batch_size 32 --lr 0.03 --buffer_size 500 --num_workers 4"
```

### CIFAR10 Symmetric 20%

```bash
python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.2 --noise_type symm \
  --sap_scale_coff 1000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.5 \
  --ogc_buffer_penalty_coeff 1.5 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR10 Symmetric 40%

```bash
python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.4 --noise_type symm \
  --sap_scale_coff 2000 \
  --ogc_loss_weight 0.4 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR10 Symmetric 60%

```bash
python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.6 --noise_type symm \
  --sap_scale_coff 3000 \
  --ogc_loss_weight 0.5 \
  --ogc_low_conf_weight 0.2 \
  --ogc_buffer_penalty_coeff 2.5 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR10 Asymmetric 20%

```bash
python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.2 --noise_type asym \
  --sap_scale_coff 1000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.5 \
  --ogc_buffer_penalty_coeff 1.5 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### CIFAR10 Asymmetric 40%

```bash
python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.4 --noise_type asym \
  --sap_scale_coff 2000 \
  --ogc_loss_weight 0.4 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```



## Food101N

Food101N 使用真实 noisy train label，clean test 来自 Food-101。不要再设置 CIFAR 式合成噪声，正式命令保持 `--noise_rate 0`。

### 数据目录检查

先把 Food-101N noisy train 和 Food-101 clean test 解压到 `data/` 下。推荐先运行纯数据检查脚本：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N_release \
  --test-list ../food-101/meta/test.txt \
  --test-images-dir ../food-101/images \
  --classes-file ../food-101/meta/classes.txt \
  --strict
```

如果本地文件名不同，把 `--test-list`、`--test-images-dir`、`--classes-file` 改成真实路径。若 train metadata 存在，也可以显式加：

```bash
--train-list meta/train.txt --train-images-dir train
```

项目环境中已安装 `torch`、`torchvision` 后，可以进一步检查 Mammoth 数据集类是否能构造 loader：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N_release \
  --test-list ../food-101/meta/test.txt \
  --test-images-dir ../food-101/images \
  --classes-file ../food-101/meta/classes.txt \
  --strict \
  --check-import
```

如果希望同时真实读取一个 train/test batch，再加 `--check-batch`。

### Food101N Debug Smoke

只用于验证数据、模型、loader、buffer、OGC/SAP 调用链是否能跑通，不作为论文结果：

```bash
python main.py --dataset seq-food101n --model ogc-sap --enable_sap 1 \
  --food101n_root data/Food-101N_release \
  --food101n_test_list ../food-101/meta/test.txt \
  --food101n_test_images_dir ../food-101/images \
  --food101n_classes_file ../food-101/meta/classes.txt \
  --backbone resnet18 --n_epochs 1 --batch_size 8 --minibatch_size 8 \
  --lr 0.03 --buffer_size 500 --num_workers 0 --debug_mode 1 \
  --noise_rate 0 --seed 0
```

### Food101N 正式训练模板

真实实验建议先从 OGC+SAP 主方法开始，确认显存后再调大 batch 或 backbone：

```bash
COMMON_ARGS="--dataset seq-food101n --food101n_root data/Food-101N_release --food101n_test_list ../food-101/meta/test.txt --food101n_test_images_dir ../food-101/images --food101n_classes_file ../food-101/meta/classes.txt --backbone resnet18 --n_epochs 50 --batch_size 32 --minibatch_size 32 --lr 0.03 --buffer_size 2000 --num_workers 4 --noise_rate 0"

python main.py --model ogc-sap --enable_sap 1 \
  --sap_scale_coff 5000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 2000 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

可对照消融：

```bash
python main.py --model er-ace-aer-abs --savecheck last --seed 0 ${COMMON_ARGS}
python main.py --model aer-sap --enable_sap 1 --sap_retain_samples 2000 --savecheck last --seed 0 ${COMMON_ARGS}
python main.py --model ogc-sap --enable_sap 0 --ogc_loss_weight 0.3 --ogc_low_conf_weight 0.3 --ogc_buffer_penalty_coeff 2.0 --savecheck last --seed 0 ${COMMON_ARGS}
```

## Ablation Commands

以下命令用于保持同一组超参数，只切换模块组合。

### Baseline: AER/ABS

```bash
python main.py --dataset seq-cifar100 --model er-ace-aer-abs --noise_rate 0.6 --noise_type symm \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### SAP Only

```bash
python main.py --dataset seq-cifar100 --model aer-sap --noise_rate 0.6 --noise_type symm \
  --sap_scale_coff 5000 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### OGC Only

```bash
python main.py --dataset seq-cifar100 --model ogc-sap --enable_sap 0 --noise_rate 0.6 --noise_type symm \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

### OGC + SAP

```bash
python main.py --dataset seq-cifar100 --model ogc-sap --enable_sap 1 --noise_rate 0.6 --noise_type symm \
  --sap_scale_coff 5000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```
## NTU60

NTU60 推荐 `buffer_size=500`。

```bash
COMMON_ARGS="--n_epochs 30 --batch_size 32 --lr 0.1 --buffer_size 500"
```

### NTU60 Symmetric 40%
```bash
python main.py \
  --dataset seq-ntu60 \
  --model aer_ogc_sap \
  --noise_rate 0.4 \
  --noise_type symmetric \
  --seed 0 
```
