# Food101N Phase 3 Implementation Design

日期：2026-07-14

## 目标

在当前 Mammoth 项目中实现最小、可注册、可 dry-run 验证的 Food101N 数据集支持。实现只覆盖数据集接入，不启动完整训练，不修改噪声鲁棒算法。

## 文件改动设计

| 文件 | 类型 | 职责 |
|---|---|---|
| `datasets/seq_food101n.py` | 修改 | 实现 raw image dataset、metadata 解析、PuriDivER JSON 兼容、Food-101 clean test 兼容、continual wrapper |
| `datasets/configs/seq-food101n/default.yaml` | 新增 | 声明 Food101N 默认 task split、尺寸、均值方差、训练默认参数 |
| `docs/food101n/01_project_dataset_audit.md` | 新增 | 项目结构调查 |
| `docs/food101n/02_food101n_web_research.md` | 新增 | 官方来源、候选实现、许可证和协议研究 |
| `docs/food101n/03_food101n_implementation_design.md` | 新增 | 本设计 |
| `docs/food101n/04_food101n_validation_report.md` | 新增 | 最小验证结果 |
| `AGENTS.md` | 修改 | 按项目规则记录新增数据集和待确认项 |

不修改 `datasets/__init__.py`，因为动态扫描已足够注册。

## Dataset 类结构

新增底层 dataset：

```text
Food101N(Dataset)
```

职责：

1. 只按需从磁盘读取图像，不把图片加载进内存。
2. 暴露 `data` 和 `targets`，供 `MammothDatasetWrapper` 过滤。
3. 维护 `classes`、`class_to_idx`、`idx_to_class`。
4. 对 train 使用 noisy class labels。
5. 对 test 使用 clean Food-101 labels。
6. 记录可选 `verification_labels`，但不把 verification label 当 clean class label。
7. 支持 `target_transform`。
8. `__getitem__` 返回 `(img, target, not_aug_img)`。

新增 continual wrapper：

```text
SequentialFood101N(ContinualDataset)
```

职责：

1. `NAME = 'seq-food101n'`
2. `SETTING = 'class-il'`
3. `N_CLASSES = 101`
4. `N_TASKS = 5`
5. `N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]`
6. 构建 train/test raw datasets。
7. 调用 `store_masked_loaders()`，保留项目原有 task split、test loader 累积、DataLoader seed 逻辑。
8. `get_class_names()` 返回随 class order 修正后的类名。

## 支持的 metadata 格式

实现应支持三类输入，以适配官方数据和公开 continual split：

### 1. PuriDivER task JSON

字段：

```text
file_name
klass
label
verification_label
```

说明：

1. `label` 是 noisy/current protocol 的 class id。
2. `verification_label` 是二值验证标签，不是 clean class id。
3. `klass` 可用于建立或校验 class name。

### 2. Food-101 JSON

格式类似：

```json
{
  "class_name": ["class_name/image_id", "..."]
}
```

说明：

1. 用于 Food-101 clean test metadata。
2. 类别排序采用稳定排序。
3. image path 解析为 `<images_dir>/<class>/<image_id>.jpg`。

### 3. CSV/TSV/TXT list

支持常见字段：

```text
file_name / filepath / path / image
klass / class / class_name / category
label / target
verification_label / verified
```

说明：

1. 如果 label 缺失但 class name 存在，用 `class_to_idx` 映射。
2. 如果只有路径如 `class_name/image.jpg`，从路径第一段推断 class name。
3. 文件排序保持 metadata 顺序；目录扫描 fallback 使用 sorted order。

## 数据路径策略

不自动下载 Food101N。默认 root 探测：

```text
<base_path>/Food-101N
<base_path>/Food-101N_release
<base_path>/food-101n
<base_path>/food-101
```

推荐显式传参：

```bash
--food101n_root /path/to/Food-101N_release
--food101n_train_list /path/to/train.json
--food101n_test_list /path/to/food-101/meta/test.json
--food101n_images_dir /path/to/images_or_train
```

路径参数如果是相对路径，则相对 `--base_path` 解析；绝对路径原样使用。

## train/test 设计

| split | 图像来源 | label 来源 | label 语义 |
|---|---|---|---|
| train | Food-101N noisy training images | noisy class label，常见为目录名或 metadata `label` | noisy train target |
| test | Food-101 clean test set | Food-101 test metadata 或目录名 | clean evaluation target |

必须保证 train/test 使用同一个 `class_to_idx`。如果官方 `classes.txt` 或 Food-101 meta 可用，优先使用；否则从 train/test metadata 类名并集排序得到。

## 类增量任务设计

采用自然稳定类别顺序：

```text
sorted(class_names)
```

任务边界：

```text
[0, 20, 40, 60, 80, 101]
```

原论文和官方 Food101N 页面未公开 class-incremental class order，因此不得声称本实现使用“原论文 class order”。如需复现某篇 continual/noisy paper 的顺序，应使用 `--custom_class_order` 或 PuriDivER task JSON。

## transforms

采用 ImageNet 风格 224 输入：

```text
train: Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip -> ToTensor -> Normalize(ImageNet)
test:  Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
```

依据：

1. Food101N 是自然图像，官方 baseline 使用 ResNet-50。
2. Food-101/Food101N 图像不是 CIFAR 32x32。
3. 当前项目已有 224 自然图像数据集使用 ImageNet normalization。

默认 backbone 设为 `resnet18`，避免自动下载预训练权重；用户后续可显式指定其他 backbone。

## true_labels 与 verification label

本实现不伪造 clean train class labels。Food101N train 的 `targets` 是 noisy class labels。`verification_labels` 如存在，仅保存在 dataset 属性中，表示 noisy label 是否人工验证为正确。

当前部分模型如 `er-ace-aer-abs` 强制要求 `true_labels` 参数，这是算法接口与真实 Food101N 协议之间的后续问题。本阶段不修改算法；如果后续要跑这些模型，需要用户确认是否：

1. 修改模型签名允许 `true_labels=None`；
2. 只使用支持 `true_labels=None` 的模型；
3. 单独设计真实噪声数据的兼容字段并清楚标注语义。

Food101N 训练标签已经是真实 noisy class label，因此本数据集应拒绝 `--noise_rate > 0`，避免再次触发 CIFAR 式 synthetic noise 注入。

## 非均匀 task 兼容

当前框架主路径已支持 list task size。Food101N 本阶段使用：

```python
N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]
```

不修改通用训练器。已知风险：半监督 `label_perc != 1` 路径暂不支持 list task size，本阶段不覆盖该实验。

## 测试计划

1. `python3 -m py_compile datasets/seq_food101n.py`
2. 在缺少 torch 的环境中记录 import 测试未执行。
3. 用纯 Python/AST 或 mock fixture 验证文件存在、task split 常量、文档和配置。
4. 若可用 Python 环境装有 torch/torchvision，则创建临时小型图片和 metadata，验证注册、单样本、单 batch。
5. 检查 `git diff`，确认未改算法和 `readme_latest.md`。

## 风险与待确认项

1. 本机没有真实 Food101N 数据，无法验证官方样本数量。
2. Kaggle 数据包内部文件名需用户下载后确认。
3. 原 Food101N 官方没有 class-incremental class order。
4. AER/BMVC 2024 具体 Food101N transform/epoch 等需后续人工核对 PDF。
5. `er-ace-aer-abs` 对 `true_labels` 的强制需求不适合没有 clean train class labels 的真实 Food101N，本阶段不改算法。
