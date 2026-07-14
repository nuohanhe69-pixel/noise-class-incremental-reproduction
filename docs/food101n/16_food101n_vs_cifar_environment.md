# Food101N 实验环境与 CIFAR10/100 的差异

## 简短结论

Food101N 不需要单独换一套 Python/Conda 环境；它仍然在当前 Mammoth 复现项目里运行，主要依赖还是 PyTorch、torchvision、PIL 等。

但 Food101N 的实验配置、数据读取方式、输入尺寸、模型 backbone、噪声处理和计算资源需求都和 CIFAR10/100 明显不同。不能简单把 CIFAR10/100 的命令换成 `--dataset seq-food101n` 就直接认为环境完全一致。

## 当前 Mammoth 项目中的主要差异

| 项目 | CIFAR10 | CIFAR100 | Food101N |
|---|---:|---:|---:|
| 数据入口 | `seq-cifar10` | `seq-cifar100` | `seq-food101n` |
| 数据来源 | torchvision CIFAR10 自动下载/读取 | torchvision CIFAR100 自动下载/读取 | 本地 Food-101N 图片 + metadata |
| 图片格式 | 内置 batch 数据，加载后转 PIL | 内置 batch 数据，加载后转 PIL | 真实 jpg/png 图片路径 |
| 输入尺寸 | 32x32 | 32x32 | 224x224 |
| 类别数 | 10 | 100 | 101 |
| 增量任务 | 5 tasks x 2 classes | 常用 5 tasks x 20 classes 或 20 tasks x 5 classes | 5 tasks: `[20, 20, 20, 20, 21]` |
| 默认/正式 backbone | ResNet18 | 当前仓库默认 ResNet18，NTD 参考为 ResNet32 | 当前正式配置为 ResNet34 |
| 当前项目 epoch 配置 | 默认 50 | 默认 50 | 当前正式配置 20 |
| batch size | 32 | 32 | 32 |
| minibatch size | 视命令/模型而定 | 视命令/模型而定 | 当前正式配置 32 |
| 噪声来源 | 通常通过 `--noise_rate` 人工注入 | 通常通过 `--noise_rate` 人工注入 | 数据集自带真实 noisy label |
| 是否应额外加 `--noise_rate` | 可以，按实验设置 | 可以，按实验设置 | 不应加；应保持 `noise_rate=0` |
| test set | CIFAR 官方 clean test | CIFAR 官方 clean test | 当前采用 NTD/PuriDivER split 的 4,741 test |
| normalization | CIFAR mean/std | CIFAR100 mean/std | 当前项目使用 ImageNet mean/std |

## 数据处理差异

### CIFAR10/100

CIFAR10/100 是 torchvision 直接支持的数据集。项目中的 `datasets/seq_cifar10.py` 和 `datasets/seq_cifar100.py` 会自动使用：

```text
base_path()/CIFAR10
base_path()/CIFAR100
```

如果本地没有数据，torchvision 会尝试下载。数据本身是 32x32 小图，标签是 clean label。噪声实验一般通过命令参数注入人工噪声，例如 symmetric noise 或 asymmetric noise。

### Food101N

Food101N 不是 torchvision 内置的 Food101。Food101N 是 noisy web image dataset，需要本地真实图片文件和 metadata：

```text
data/Food-101N/meta/train.tsv
data/Food-101N/meta/test.tsv
data/Food-101N/meta/classes.txt
data/Food-101N/images/
```

当前项目已经生成的是 metadata，也就是 52,867 train / 4,741 test 的图片清单。真实图片还需要下载后放到或链接到 `images/` 下。

Food101N 的训练标签本身就是 noisy label，`verification_label` 是人工验证标记。因此 Food101N 不能再像 CIFAR 那样额外加合成噪声。

## 图像变换差异

CIFAR10/100 使用 32x32 小图增强：

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip()
ToTensor()
Normalize(CIFAR mean/std)
```

Food101N 使用 224x224 图像增强：

```text
Resize(256, bicubic)
RandomCrop(224)
RandomHorizontalFlip()
ToTensor()
Normalize(ImageNet mean/std)
```

Food101N test transform 是：

```text
Resize(256, bicubic)
CenterCrop(224)
ToTensor()
Normalize(ImageNet mean/std)
```

这意味着 Food101N 的显存和训练时间会明显高于 CIFAR10/100。

## backbone 与训练配置差异

当前项目正式 Food101N 配置在：

```text
datasets/configs/seq-food101n/default.yaml
```

关键配置是：

```yaml
N_CLASSES: 101
N_TASKS: 5
N_CLASSES_PER_TASK: [20, 20, 20, 20, 21]
SIZE: [224, 224]
MEAN: [0.485, 0.456, 0.406]
STD: [0.229, 0.224, 0.225]
backbone: resnet34
n_epochs: 20
batch_size: 32
minibatch_size: 32
```

需要注意：`datasets/seq_food101n.py` 里的代码 fallback 仍然返回 `resnet18` 和 `50` epoch，但正式运行时应以 YAML 或命令行覆盖为准，即使用 `resnet34` 和当前确定的 Food101N epoch 配置。

## NTD 参考环境中的差异

如果参考 NTD 源码，Food101N 与 CIFAR10/100 的设置也不同：

| 数据集 | tasks | classes per task | memory size | model | epoch 设置 | batch |
|---|---:|---:|---:|---|---:|---:|
| CIFAR10 | 5 | 2 | 500 | ResNet18 | 255 | 16 |
| CIFAR100 | 5 | 20 | 2000 | ResNet32 | 255 | 16 |
| Food-101N | 5 | 20 | 2000 | ResNet34 | 127 | 16 |

NTD 里 Food-101N 的输入尺寸也是 224，而 CIFAR10/100 是 32。

## 对运行机器的影响

Food101N 比 CIFAR 更吃资源，主要原因是：

1. 图片从 32x32 变成 224x224。
2. backbone 从 ResNet18/ResNet32 级别变为 ResNet34。
3. 数据从内置小图变为磁盘 jpg 读取，I/O 更重。
4. 当前 split 虽然只有 57,608 张，但每张图更大。

因此 Food101N 推荐使用 GPU 跑。若显存不足，优先降低：

```text
--batch_size
--minibatch_size
--num_workers
```

但如果要做正式对比实验，batch/minibatch 改动需要在结果报告里说明。

## 正式复现时建议保持的口径

Food101N 和 CIFAR10/100 使用同一个项目环境和同一套算法入口，但它不是 CIFAR 的普通替换数据集。正式实验应说明：

```text
Food101N 使用真实噪声标签、224x224 输入、ResNet34、5 个增量任务 `[20,20,20,20,21]`、52,867/4,741 的 NTD/PuriDivER split，并且不额外注入 synthetic noise。
```

