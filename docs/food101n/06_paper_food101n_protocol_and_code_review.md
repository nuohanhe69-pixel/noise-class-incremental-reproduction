# Food101N 原文处理方式与本仓库实现说明

日期：2026-07-14  
对应原文：`Noise-Robust Class-Incremental Learning via Dynamic Gradient Constraint and Cumulative Bias Projection.pdf`

## 结论先行

原文将 Food101N 作为真实世界噪声标签数据集使用，而不是像 CIFAR 那样再人工注入 symmetric/asymmetric synthetic noise。Food101N 的增量协议是 101 类分成 5 个 class-incremental tasks：前 4 个 task 每个 20 类，最后 1 个 task 21 类，任务之间类别互斥。训练标签直接使用 Food101N 的 noisy class label；测试使用 clean Food-101 test labels；评价指标是训练完所有任务后的 Final Average Accuracy, FAA。

本仓库当前 `seq-food101n` 的核心设计与这个协议一致：`N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]`，拒绝 `--noise_rate > 0`，保留 `verification_labels` 但不把它当 clean class label，测试集通过 Food-101 clean metadata 进入同一个 class mapping。

当前已经按用户确认的 Food101N 复现协议更新项目配置：

1. Food101N 采用论文/AER/NTD 接近的 `52,867 train / 4,741 test` split，不直接扫描完整约 310k Food101N 作为首轮正式复现。
2. `datasets/configs/seq-food101n/default.yaml` 已改为 `resnet34`、20 epochs、batch size 32。
3. `readme_latest.md` 和验证脚本采用 `buffer_size=2000` 作为 AER/NTD 风格项目协议；但原文主文只写 Food101N memory follows AER，没有在主文单独明示该数值，因此报告中仍应说明依据来自 AER/NTD 风格设置。

## 我阅读和参考的材料

### 论文

| 论文 | 本次使用方式 | 链接或位置 |
|---|---|---|
| Noise-Robust Class-Incremental Learning via Dynamic Gradient Constraint and Cumulative Bias Projection | 用户提供的项目原文；用于确认 Food101N 协议、训练设置和结果表 | 本地 PDF：`/Users/hunk/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_i1ps5n9ndiyk22_8782/temp/RWTemp/2026-06/9e20f478899dc29eb19741386f9343c8/Noise-Robust Class-Incremental Learning via Dynamic Gradient Constraint and Cumulative Bias Projection.pdf` |
| CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise, CVPR 2018 | Food101N 数据集来源；确认 101 类、约 310k web images、noisy class labels、verification labels、Food-101 test evaluation | https://kuanghuei.github.io/Food-101N/ ，https://kuanghuei.github.io/CleanNetProject/ |
| Food-101: Mining Discriminative Components with Random Forests, ECCV 2014 | Food-101 clean test 来源；确认 101 类，每类 250 manually reviewed test images | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ |
| Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries, CVPR 2022 | PuriDivER；确认真实噪声 continual setting、Food-101N 使用、公开代码/数据组织线索 | https://arxiv.org/abs/2203.15355 ，https://github.com/clovaai/puridiver |
| May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels, BMVC 2024 | AER/ABS baseline；确认 AER 的 alternate replay、Food-101N 5 tasks、ResNet34、20 epochs、batch size 32 等设置 | https://arxiv.org/abs/2408.14284 ，https://bmvc2024.org/proceedings/680/ |
| Deep Residual Learning for Image Recognition, CVPR 2016 | Food101N backbone 使用 ResNet-34 的依据 | https://arxiv.org/abs/1512.03385 |

### 源码和项目页

| 源码/项目 | 本次使用方式 | 链接或本地路径 |
|---|---|---|
| 本仓库 Food101N dataset | 当前 Food101N 数据处理实现主体 | `datasets/seq_food101n.py` |
| 本仓库 Food101N validation script | 当前数据完整性检查脚本 | `scripts/validate_food101n_dataset.py` |
| Mammoth continual loader | 解释 class split、task mask、test loader 累积机制 | `datasets/utils/continual_dataset.py` |
| 本仓库 AER/ABS baseline | 对照 `true_labels`、buffer insertion、AER alternate schedule | `models/er_ace_aer_abs.py` |
| 本仓库 OGC+SAP | 对照 DGC/OGC、SAP/CBP 工程实现入口 | `models/ogc_sap.py` |
| 本仓库 AER-SAP | 对照 SAP-only 消融 | `models/aer_sap.py` |
| upstream Mammoth AER source | 确认 `er_ace_aer_abs.py` 来自 Mammoth，上游原始签名强制 `true_labels`，但 observe 内部不使用它 | https://github.com/aimagelab/mammoth |
| torchvision Food101 | 参考 Food-101 metadata 读取方式：`meta/train.json`/`meta/test.json`，类别名 sorted 后映射到 id | https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py |
| PuriDivER official repo | 参考数据目录格式、Food-101N 作为真实噪声数据集、`blurry10` 真实噪声协议线索；因 GPL-3.0 许可证，本仓库不复制其实现 | https://github.com/clovaai/puridiver |
| CleanNet TensorFlow code | 只参考数据字段语义和 Food101N 来源，不复制代码 | https://github.com/kuanghuei/clean-net |
| cleanlab label-error-detection benchmark | 参考 Kaggle 解压路径和 `verified_train.tsv` 线索 | https://github.com/cleanlab/label-error-detection-benchmarks |

## 原文中 Food101N 是怎么处理的

### 数据集语义

Food101N 在原文中被当作 real-world noisy dataset。原因是其图片从 web 自动收集，标签带有更真实、更不规则的 instance-dependent noise。论文明确区别了两类噪声设置：

| 数据集 | 噪声来源 |
|---|---|
| CIFAR-10 / CIFAR-100 | 人工注入 symmetric/asymmetric synthetic noise |
| NTU-60 | 人工注入 symmetric synthetic noise |
| Food101N | 不注入 synthetic noise，直接使用原始真实 noisy labels |

这意味着 Food101N 不能再设置 `--noise_rate 0.2/0.4/0.6`。如果再注入一次 synthetic noise，就会变成“真实噪声 + 合成噪声”的混合协议，不再对应原文。

### 类增量任务划分

原文 Food101N 的 class-incremental split：

| 项目 | 设置 |
|---|---|
| 总类别数 | 101 |
| task 数 | 5 |
| 每 task 类别数 | `[20, 20, 20, 20, 21]` |
| task 关系 | 类别互斥 |
| 旧数据访问 | 进入新 task 后，之前 task 的原始训练样本不可访问 |
| 旧知识保持 | 只保留 fixed-capacity replay buffer |

对应边界是：

```text
Task 0: labels [0, 20)
Task 1: labels [20, 40)
Task 2: labels [40, 60)
Task 3: labels [60, 80)
Task 4: labels [80, 101)
```

本仓库通过 `SequentialFood101N.N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]` 表达这个非均匀 split。Mammoth 的 `get_offsets()` 已支持 list 类型，因此不需要改通用训练器。

### 训练和评价设置

原文主文给出的 Food101N 设置：

| 项目 | 原文设置 |
|---|---|
| noise | 使用真实 noisy labels，不注入 synthetic noise |
| backbone | ResNet-34 |
| epochs | 每 task 20 epochs |
| batch size | 32 |
| optimizer | SGD |
| memory buffer | follow AER for fair comparison，主文未明示具体数值 |
| DGC queues | 长度 2048 |
| DGC warm-up | 每 task 开始 10 epochs |
| DGC lazy update | 每 10 iterations 更新一次阈值 |
| low-confidence weight | `omega_low = 0.2` |
| buffer admission penalty | `beta = 2.0` |
| evaluation | Final Average Accuracy, FAA |

AER 论文补充说明 Seq. Food-101N 使用 5 tasks、ResNet34、20 epochs、batch size 32，且训练图片是 224x224。它还提到 Food-101N split 中每类平均约 523 张图、总数约 52,867，这接近 Food101N 官方 verification-labeled train subset 的规模，而不是完整约 310k 张的 Food101N 全量图片。因此若目标是严格复现 AER/原文表格，必须确认使用的是作者的 Food101N task metadata/split，而不是随手扫描完整 `Food-101N_release/train`。

### 原文结果表

原文 Table III 的 Food101N FAA：

| Method | Food101N FAA |
|---|---:|
| Joint | 39.91 ± 1.05 |
| PuriDivER.ME | 28.62 ± 0.85 |
| AER | 29.86 ± 1.18 |
| Ours | 31.62 ± 0.98 |

这里的 `Ours` 是 DGC + CBP 的完整方法。它比 AER 高 1.76 个百分点，比 PuriDivER.ME 高 3.00 个百分点。

## 当前 Food101N 代码是怎么实现的

### 文件入口

核心文件：

```text
datasets/seq_food101n.py
```

辅助检查脚本：

```text
scripts/validate_food101n_dataset.py
```

配置文件：

```text
datasets/configs/seq-food101n/default.yaml
```

### 底层数据集 `Food101N`

`Food101N(Dataset)` 是一个普通 PyTorch Dataset，职责是把本地图片和 metadata 转成 Mammoth 可以切 task 的结构。

关键属性：

| 属性 | 含义 |
|---|---|
| `self.data` | 图片路径数组，不提前读入图片 |
| `self.targets` | class label；train 是 noisy label，test 是 clean label |
| `self.classes` | 类名列表 |
| `self.class_to_idx` | 类名到 label id 的映射 |
| `self.idx_to_class` | label id 到类名 |
| `self.verification_labels` | 可选 verification flag；缺失为 -1 |
| `self.noisy_targets` | train split 下保存 noisy targets |
| `self.clean_targets` | test split 下保存 clean targets |

为什么这样做：

1. Food101N 图片规模较大，不能一次性读进内存，所以 `self.data` 只保存路径，`__getitem__` 时再用 PIL 读取。
2. Mammoth 的 `MammothDatasetWrapper` 要求 dataset 暴露 `data` 和 `targets`，否则不能按 class mask 切任务。
3. Food101N train 没有完整 clean class labels，因此 `targets` 必须是 noisy class labels，不能伪造 clean labels。
4. verification label 不是类别标签，只表示“当前 noisy class label 是否正确”，所以只保存，不参与训练标签替换。

### metadata 兼容

实现支持多种 metadata，是为了适配 Food101N 官方包、Food-101、PuriDivER/AER split 和用户手工整理目录。

支持字段：

| 类型 | 支持字段 |
|---|---|
| 图片路径 | `file_name`, `filepath`, `file_path`, `path`, `image`, `img`, `sample_key` |
| 类名 | `klass`, `class`, `class_name`, `category`, `category_name`, `label_name` |
| 类别 id | `label`, `target`, `class_idx`, `class_id`, `category_id` |
| verification | `verification_label`, `verified`, `is_verified` |

支持文件格式：

| 格式 | 处理方式 |
|---|---|
| JSON list | 每个 item 可以是 dict 或 path string |
| JSON dict with `annotations/samples/data/images/items` | 读取对应 list |
| Food-101 style JSON dict | `{class_name: [image_id, ...]}` |
| TXT/CSV/TSV | 自动识别 header，或按 `path label/class verification` 顺序解析 |
| class folder | 如果没有 metadata，扫描 `<images_dir>/<class_name>/<image>` |

为什么这样做：

1. Food101N 官方 Kaggle 包、Food-101 官方 metadata、PuriDivER split 的字段不完全一致。
2. PuriDivER 是 GPL-3.0，不能复制其 Dataset 代码；当前实现只兼容公开字段语义。
3. 用户本机数据路径未知，所以需要允许显式传 `--food101n_train_list`、`--food101n_test_list`、`--food101n_classes_file`。

### 路径解析

`SequentialFood101N.__init__()` 支持这些参数：

```text
food101n_root
food101n_train_list
food101n_test_list
food101n_images_dir
food101n_train_images_dir
food101n_test_images_dir
food101n_classes_file
food101n_strict
```

默认 root 候选：

```text
<base_path>/Food-101N
<base_path>/Food-101N_release
<base_path>/food-101n
<base_path>/Food101N
<base_path>/food101n
```

默认 train images 候选：

```text
train
Food-101N_release/train
images
.
```

默认 test images 候选：

```text
test
val
images
Food-101/images
food-101/images
.
```

为什么这样做：

1. Food101N train 和 Food-101 test 经常来自不同压缩包。
2. Kaggle/官方/第三方整理后的目录名可能不同。
3. `--food101n_strict 1` 可以让路径缺失立刻失败，适合正式实验前检查。

### 类别映射

代码优先使用 `food101n_classes_file`。如果没有，则从 metadata 或目录名推断：

1. 如果 metadata 中有 label 和 class name，校验二者一致。
2. 如果只有 class name，则按稳定排序生成 `class_to_idx`。
3. 如果只有 label，则生成 `food_class_000 ... food_class_100`。
4. test dataset 会复用 train dataset 的 `class_to_idx`，保证 train/test id 对齐。

为什么这样做：

1. Food101N 和 Food-101 必须共享 101 类映射，否则 train label 和 test label 会错位。
2. torchvision 的 `Food101` 也是从 metadata key sorted 后建立类别映射，这是一种稳定、可复现的做法。
3. 原文没有公开 class order，因此当前实现不能声称“使用原文 class order”。严格复现时应拿到作者 split 或显式传 `--custom_class_order`。

### 图像 transform

当前代码：

```text
train: Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip -> ToTensor -> Normalize(ImageNet)
test:  Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
not_aug: Resize(256) -> CenterCrop(224) -> ToTensor
```

为什么这样做：

1. Food101N 是自然图像，不是 32x32 CIFAR。
2. ResNet 系列通常使用 ImageNet mean/std。
3. AER 论文说明对 stream 和 buffer 使用 random crops 和 horizontal flips，Food-101N 图片为 224x224。
4. `not_aug` 输入保留未随机增强版本，用于 AER/OGC 的 loss-based sample selection 和 buffer score。

### Continual wrapper `SequentialFood101N`

关键常量：

```python
NAME = "seq-food101n"
SETTING = "class-il"
N_CLASSES = 101
N_TASKS = 5
N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]
SIZE = (224, 224)
MEAN, STD = ImageNet mean/std
```

`get_data_loaders()` 做的事情：

1. 检查 `noise_rate`，只允许 0。
2. 构建 train `Food101N`，其 labels 是真实 noisy labels。
3. 检查 train classes 必须为 101。
4. 构建 test `Food101N`，复用 train `class_to_idx`。
5. 调用 `store_masked_loaders(train_dataset, test_dataset, self)`。

`store_masked_loaders()` 的作用：

1. 每调用一次，`setting.c_task += 1`。
2. 根据当前 task 的 `[start_c, end_c)` 切 train/test。
3. 训练 loader 只包含当前 task 的类。
4. test loader 会 append 到 `dataset.test_loaders`，训练后可以评估已见任务。

这正好对应 class-incremental learning：每个 task 只训练新类，但最终要评估所有已学类。

### `true_labels` 兼容

Mammoth 的合成噪声路径会在 `noise_rate > 0` 时保存 clean targets 到 extra field `true_labels`，然后把 noisy targets 传给模型。Food101N 不走这条路径，因为它的 train labels 已经是真实 noisy labels，而且没有完整 clean train class label。

因此当前仓库做了两个处理：

1. `seq_food101n.py` 拒绝 `--noise_rate > 0`。
2. `models/er_ace_aer_abs.py` 的 `observe(..., true_labels)` 改成 `observe(..., true_labels=None)`。

这个修改是合理的，因为 upstream Mammoth 的 `er_ace_aer_abs.observe()` 虽然签名强制要求 `true_labels`，但函数内部并没有使用它。`ogc_sap.observe()` 本来就支持 `true_labels=None`，只有在存在 clean labels 时才做 noisy/clean 对照日志或 buffer 保存。

## 数据检查脚本实现

文件：

```text
scripts/validate_food101n_dataset.py
```

它是训练前检查入口，避免等到训练跑起来才发现路径或 label mapping 错。

检查内容：

| 检查 | 说明 |
|---|---|
| root | 展开相对/绝对路径 |
| metadata | 支持 JSON/TXT/CSV/TSV 和 class-folder scan |
| class count | 默认必须是 101 |
| label range | 默认必须是 0..100 |
| image paths | 默认统计缺失，`--strict` 下直接失败 |
| task split | 检查 `[20, 20, 20, 20, 21]` sum 为 101 |
| Mammoth import | `--check-import` 时实例化 `SequentialFood101N` |
| batch read | `--check-import --check-batch` 时读取一个 train/test batch |

推荐命令：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N_release \
  --test-list ../food-101/meta/test.txt \
  --test-images-dir ../food-101/images \
  --classes-file ../food-101/meta/classes.txt \
  --strict
```

如果项目环境中有 `torch` 和 `torchvision`：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N_release \
  --test-list ../food-101/meta/test.txt \
  --test-images-dir ../food-101/images \
  --classes-file ../food-101/meta/classes.txt \
  --strict \
  --check-import \
  --check-batch
```

## 本仓库当前与原文设置的对齐程度

| 项目 | 原文 | 当前仓库 | 状态 |
|---|---|---|---|
| dataset name | Food101N / Seq. Food-101N | `seq-food101n` | 已对齐 |
| class count | 101 | 101 | 已对齐 |
| task split | `[20, 20, 20, 20, 21]` | `[20, 20, 20, 20, 21]` | 已对齐 |
| noise | real noisy labels, no synthetic noise | 拒绝 `--noise_rate > 0` | 已对齐 |
| train label | noisy class label | noisy class label | 已对齐 |
| test label | Food-101 clean test | 支持 Food-101 test metadata/images | 已对齐 |
| verification label | 不是 clean class label | 只保存，不替换 targets | 已对齐 |
| backbone | ResNet-34 | 默认 ResNet-34 | 已对齐 |
| epochs | 20 | 默认 20 | 已对齐 |
| batch size | 32 | 默认 32 | 已对齐 |
| buffer size | follow AER，主文未明示 | 项目协议采用 2000 | 按 AER/NTD 风格对齐；原文主文未单独明示 |
| Food101N subset | AER 文本显示约 52,867 train / 4,741 test | 项目已确认采用 52,867 / 4,741 split | 已确认 |

## 严格复现原文 Food101N 的建议命令

在数据 split 已确认后，应优先用原文设置：

```bash
COMMON_ARGS="--dataset seq-food101n \
  --food101n_root data/Food-101N \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 \
  --n_epochs 20 \
  --batch_size 32 \
  --minibatch_size 32 \
  --lr 0.03 \
  --noise_rate 0 \
  --num_workers 4"

python main.py --model ogc-sap --enable_sap 1 \
  --buffer_size 2000 \
  --ogc_queue_size 2048 \
  --ogc_warmup_epochs 10 \
  --ogc_lazy_update 10 \
  --ogc_low_conf_weight 0.2 \
  --ogc_buffer_penalty_coeff 2.0 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

如果暂时没有作者配置，可以先用工程 smoke test：

```bash
python main.py --dataset seq-food101n --model ogc-sap --enable_sap 1 \
  --food101n_root data/Food-101N \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 --n_epochs 1 --batch_size 8 --minibatch_size 8 \
  --lr 0.03 --buffer_size 500 --num_workers 0 --debug_mode 1 \
  --noise_rate 0 --seed 0
```

## 尚未完全解决的问题

1. 真实 Food101N/Food-101 数据当前本机仍未发现，所以没有跑真实 batch 和训练。
2. 项目当前 shell 没有可用 `torch`、`torchvision` 环境，所以 `--check-import --check-batch` 尚未在真实环境验证。
3. 原文说 follow AER memory-buffer setting，但没有在主文给出 Food101N 的具体 buffer size；当前项目采用 AER/NTD 风格 `buffer_size=2000`，报告中应注明这一点。
4. AER/NTD 风格 split 已确认采用约 `52,867 train / 4,741 test`；如果直接扫描完整 Food101N 约 310k train images，结果不可与原文表格直接比较。

## 后续建议

1. 获取或整理 `52,867 train / 4,741 test` 的 Food101N metadata，优先转成 `train.tsv/test.tsv/classes.txt`。
2. 准备真实数据后先跑：

```bash
python scripts/validate_food101n_dataset.py --strict --check-import --check-batch ...
```

3. 再跑 1 epoch debug smoke，最后跑正式 OGC+SAP、AER、PuriDivER.ME 对照实验。
