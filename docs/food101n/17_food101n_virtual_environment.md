# Food101N 与 CIFAR10/100 的虚拟环境差异

## 结论

Food101N 实验不需要单独创建一套新的 Python 虚拟环境。当前项目里，Food101N 和 CIFAR10/100 可以使用同一个虚拟环境。

区别主要在数据、输入尺寸、模型和显存需求，不在 Python 包环境。

## 当前项目所需依赖

当前 `requirements.txt` 中已经包含 Food101N 数据集实现需要的核心依赖：

```text
torch>=2.1.0
numpy
torchvision
Pillow
pyyaml
tqdm
timm==0.9.8
```

`datasets/seq_food101n.py` 主要使用：

```text
torch
torchvision
PIL/Pillow
numpy
csv/json/pathlib
```

其中 `csv/json/pathlib` 是 Python 标准库，不需要额外安装。

## 和 CIFAR10/100 是否要分开环境

不需要。

CIFAR10/100 也是通过 PyTorch/torchvision 读取和训练，Food101N 只是从本地图片路径读取 jpg/png，并使用 `Pillow` 打开图片。`Pillow` 已经在项目依赖里。

推荐做法是：

```bash
source <你的虚拟环境路径>/bin/activate
pip install -r requirements.txt
```

然后 CIFAR10、CIFAR100、Food101N 都在这一个虚拟环境里跑。

## 什么情况下可能需要额外包

如果只运行本项目的 Mammoth Food101N 实现，不需要额外安装 `pandas`。

但如果你要直接运行外部 NTD 仓库 `external/ntd` 的原始代码，它的 `requirements.txt` 里包含：

```text
pandas
pymongo
fire
torch_optimizer
randaugment
easydict
tensorboard
scikit-learn
matplotlib
```

这些是 NTD 原项目运行所需，不是当前 Mammoth 项目 `seq-food101n` 数据集接入必须的依赖。

## 真正需要注意的是显存和数据路径

Food101N 虽然不需要新的虚拟环境，但比 CIFAR 更吃资源：

- CIFAR10/100 是 32x32 图片。
- Food101N 是 224x224 图片。
- Food101N 当前正式配置使用 ResNet34。
- Food101N 需要本地真实图片文件，不能只靠 torchvision 自动下载。

因此 Food101N 常见问题不是虚拟环境不一样，而是：

```text
1. 图片路径没有准备好
2. 显存不够
3. batch_size / minibatch_size 过大
4. 忘记不要给 Food101N 额外设置 --noise_rate
```

## 建议口径

后续报告里可以这样写：

```text
Food101N 与 CIFAR10/100 使用同一 Python 虚拟环境和同一项目依赖；Food101N 额外差异体现在本地图片数据、224x224 输入、ResNet34 backbone、真实噪声标签和更高显存需求，而不是额外 Python 包环境。
```

