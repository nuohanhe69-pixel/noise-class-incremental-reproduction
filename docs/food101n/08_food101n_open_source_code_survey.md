# Food101N 开源数据处理代码调查

日期：2026-07-14

## 结论

不是全网没有 Food101N 相关的数据处理代码。现在能查到的公开资料里，确实存在 Food101N/Food-101N 相关代码和实验仓库。

但需要分清三类东西：

1. **Food-101 的普通分类数据加载代码**：很多，比如 torchvision `Food101`，但它处理的是 Food-101，不是 Food101N。
2. **Food101N 的普通噪声标签分类代码**：有，比如 CleanNet 官方 TensorFlow 代码和一些 noisy-label learning 仓库。
3. **Food101N 的噪声类增量/在线持续学习代码**：也有，比如 PuriDivER、NTD 这类项目，但它们的数据接口、任务协议、许可证、训练框架都和本仓库的 Mammoth 结构不同，不能直接复制粘贴使用。

所以本项目自己写 `datasets/seq_food101n.py` 的原因不是“没人开源过 Food101N 代码”，而是：现有代码没有一个完全满足本仓库的 Mammoth 数据集接口、Food101N 真实噪声语义、Food-101 clean test、101 类 5 task class-incremental split、以及当前模型训练入口的要求。

## 已查到的开源/公开来源

### 1. Food101N 官方页面与 CleanNet 官方代码

来源：

- Food101N 官方页面：https://kuanghuei.github.io/Food-101N/
- CleanNet 官方代码：https://github.com/kuanghuei/clean-net

Food101N 官方页面明确说明：

| 项目 | 信息 |
|---|---|
| 数据集 | Food-101N |
| 类别数 | 101 |
| 图片数 | 约 310,009 |
| class labels | noisy labels，每张图都有 |
| verification labels | 只标记 class label 是否正确，不是 clean class label |
| 训练 verification 数量 | 52,868 |
| 验证 verification 数量 | 4,741 |
| 分类测试 | 直接采用 Food-101 testing set |

CleanNet 仓库是官方 TensorFlow 实现，主要用于 CleanNet 论文里的 label noise learning / label noise detection。它不是 Mammoth 风格 Dataset，也不是 class-incremental learning 框架。它能提供 Food101N 的原始语义依据，但不能直接作为本项目的 `seq-food101n` 数据集文件使用。

### 2. PuriDivER 官方 PyTorch 代码

来源：

- PuriDivER：https://github.com/clovaai/puridiver

PuriDivER 是 CVPR 2022 的在线持续学习/噪声标签项目，README 里明确支持 Food-101N，并要求数据按类似下面的结构整理：

```text
[dataset name]
  train/
    [class1 name]/
    [class2 name]/
  test/
    [class1 name]/
    [class2 name]/
```

它还明确说数据集候选包含：

```text
cifar10, cifar100, WebVision-V1-2, Food-101N
```

并且对 WebVision/Food-101N 使用 `blurry10` 这类真实噪声 continual stream 设置。

这个仓库非常值得参考，但不能直接搬进本项目，主要原因是：

1. 它是 GPL-3.0 license，直接复制代码会影响本项目许可证边界。
2. 它不是 Mammoth 数据集接口。
3. 它是 online continual learning / blurry task boundary 风格，本仓库当前 Food101N 是接入 Mammoth class-incremental loader。
4. 它的数据组织更偏 class-folder，而我们需要同时兼容 Food101N noisy train、Food-101 clean test、metadata list、verification labels、AER split。

### 3. NTD 仓库

来源：

- NTD：https://github.com/wish44165/ntd

NTD 仓库也支持 Food-101N。README 中给出的 Food-101N 设置包括：

| 项目 | 数值 |
|---|---:|
| train | 52,867 |
| test | 4,741 |
| class | 101 |
| tasks | 5 |
| memory size | 2,000 |
| model | ResNet34 |

它的复现实验命令里也包含：

```bash
python run_experiment.py --dataset Food-101N \
  --dataset_path ../../../../datasets/Food-101N/images \
  --mem_manage NTD \
  --robust_type none \
  --exp_name blurry10
```

这个仓库说明 Food-101N 的 5-task continual/noisy-label setting 确实有开源实现参考。但它不是我们项目的 NRGP/DGC+CBP 复现代码，也不是 Mammoth 数据集模块。它更适合用来核对 Food101N 的样本数、memory size、ResNet34、5 tasks 等实验设置。

### 4. torchvision `Food101`

来源：

- torchvision 文档：https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101.html
- torchvision 源码：https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

torchvision 有官方 `Food101` Dataset，但它处理的是 Food-101，不是 Food101N。

它能参考的地方是：

1. Food-101 的 `meta/train.json`、`meta/test.json` 读取方式。
2. 101 类类别名映射方式。
3. clean test set 的基本结构。

它不能解决的问题是：

1. Food101N noisy train labels。
2. Food101N verification labels。
3. Food101N 与 Food-101 clean test 的组合。
4. 类增量任务划分。
5. Mammoth `data/targets` 字段和 `store_masked_loaders()` 接入。

### 5. 其他 noisy-label / label-error 代码

还能查到一些 Food101N 被用于 noisy-label learning、label-error detection 或 benchmark 的代码，例如 cleanlab 相关 benchmark、LSA/其他 noisy-label 项目等。

这些项目通常只解决普通 noisy-label classification 或 label noise detection，不解决本项目需要的 class-incremental learning。因此只能参考数据格式、transform 或实验数字，不能直接作为本项目的增量数据处理实现。

## 为什么不能直接用开源代码

本项目的 Food101N 数据处理必须同时满足这些约束：

| 约束 | 说明 |
|---|---|
| Mammoth 接口 | Dataset 必须提供 `data` 和 `targets`，并能被 `MammothDatasetWrapper` 包装 |
| class-il 设置 | 必须设置 `SETTING = "class-il"` |
| 101 类 | 必须是 `N_CLASSES = 101` |
| 5 task split | 必须是 `[20, 20, 20, 20, 21]` |
| train label | 必须使用 Food101N noisy class label |
| test label | 必须使用 Food-101 clean test label |
| verification label | 只能保存，不能当 clean class label |
| synthetic noise | Food101N 不应再额外注入 `--noise_rate` |
| metadata 兼容 | 要兼容 JSON/TXT/CSV/TSV/class-folder 等不同整理方式 |
| train/test mapping | Food101N train 和 Food-101 test 必须共享同一套 `class_to_idx` |

PuriDivER/NTD 这类代码证明“开源里有 Food101N 持续学习处理代码”，但它们并不是 Mammoth 数据集模块。torchvision 证明“开源里有 Food-101 Dataset”，但不是 Food101N。CleanNet 证明“Food101N 官方有代码”，但不是 PyTorch/Mammoth/class-incremental。

因此，本仓库写一个新的 `seq_food101n.py` 是合理的：它不是从零发明 Food101N 协议，而是把公开论文和开源项目中的数据语义，重新适配到 Mammoth 复现工程里。

## 对本项目最有参考价值的开源来源排序

| 优先级 | 来源 | 用途 |
|---:|---|---|
| 1 | PuriDivER | 核对 Food-101N 作为 noisy continual learning 数据集的公开实现和目录组织 |
| 2 | NTD | 核对 Food-101N 5 tasks、52,867 train、4,741 test、memory size 2,000、ResNet34 等设置 |
| 3 | Food101N 官方页面 | 核对 noisy label、verification label、Food-101 test evaluation 的原始语义 |
| 4 | CleanNet 官方代码 | 核对 Food101N 官方任务背景和 TensorFlow 实现语义 |
| 5 | torchvision Food101 | 参考 Food-101 clean metadata 与普通 PyTorch Dataset 写法 |

## 一句话回答

全网不是没有 Food101N 数据处理代码；有，而且 PuriDivER、NTD、CleanNet 都很有参考价值。只是目前没查到一个可以直接放进本仓库 Mammoth/NRGP 复现框架里的现成 `seq-food101n` 数据集实现，所以我们现在写的代码属于“参考公开论文和开源项目协议后，重新适配 Mammoth 的实现”。
