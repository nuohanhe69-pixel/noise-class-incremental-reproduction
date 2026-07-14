# Food101N 数据处理代码参考来源与增量设置说明

日期：2026-07-14

## 简短结论

当前 `datasets/seq_food101n.py` 这份 Food101N 数据处理代码不是直接复制某一个开源项目的 Dataset 文件，而是在 Mammoth continual learning 数据集接口上新写的兼容实现。它主要参考了 5 类来源：

1. 项目原文 `Noise-Robust Class-Incremental Learning via Dynamic Gradient Constraint and Cumulative Bias Projection`：确定 Food101N 在本文中如何作为真实噪声类增量数据集使用。
2. AER 论文 `May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels`：确定 Seq. Food-101N 的 5-task 设置、ResNet34、20 epochs、batch size 32 等复现实验设置。
3. Food101N/CleanNet 官方资料：确定 Food101N 的数据语义，也就是 web noisy train labels、verification labels 不是 clean class labels、测试沿用 Food-101 clean test set。
4. Food-101 官方数据集与 `torchvision.datasets.Food101`：参考 Food-101 的 metadata 组织方式、类别名映射和 clean test 使用方式。
5. Mammoth 开源框架和本仓库已有数据集：参考 `ContinualDataset`、`store_masked_loaders()`、`data/targets` 字段约定、class-incremental task mask 机制。

Food101N 的增量设置是 class-incremental learning：101 个类别按类别编号切成 5 个连续任务，类别数为 `[20, 20, 20, 20, 21]`。训练到某个 task 时，只给当前 task 的新类别训练数据；旧 task 的原始训练数据不能再直接访问，只能依赖 replay buffer 中保留下来的少量旧样本。模型最终要在所有已经见过的类别上做分类。

## 这里的“增量”是什么

“增量”指 continual learning / incremental learning 里的训练协议，不是普通的数据增强，也不是把数据文件一点点追加到目录里。

在普通 supervised learning 中，训练集一开始就包含全部 101 类，模型一次性看到所有类别的数据。

在 class-incremental learning 中，数据被拆成一串 task：

```text
Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4
```

模型按顺序学习。学 Task 0 时只看到 Task 0 的类别；进入 Task 1 后，只能看到 Task 1 的新类别数据，Task 0 的原始训练数据不可再随意使用。为了减少遗忘，方法通常允许保存一个固定容量的 replay buffer，里面放少量历史样本。

在本项目语境下，“噪声类增量”还多了一层含义：当前 task 的训练标签可能是 noisy labels。Food101N 本身就是 web 收集来的真实噪声标签数据，所以不再额外注入 synthetic label noise。

## Food101N 增量是怎么设置的

### 原文协议

项目原文把 Food101N 设置为 101 类、5 个 class-incremental tasks：

| Task | 类别编号范围 | 类别数 |
|---|---:|---:|
| Task 0 | `[0, 20)` | 20 |
| Task 1 | `[20, 40)` | 20 |
| Task 2 | `[40, 60)` | 20 |
| Task 3 | `[60, 80)` | 20 |
| Task 4 | `[80, 101)` | 21 |

所以总类别数是：

```text
20 + 20 + 20 + 20 + 21 = 101
```

这个设置来自项目原文和 AER 论文对 Seq. Food-101N 的实验描述。

### 本仓库代码中的设置

在 `datasets/seq_food101n.py` 中，核心设置是：

```python
class SequentialFood101N(ContinualDataset):
    NAME = "seq-food101n"
    SETTING = "class-il"
    N_CLASSES = 101
    N_TASKS = 5
    N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]
```

含义如下：

| 字段 | 含义 |
|---|---|
| `NAME = "seq-food101n"` | 命令行里使用的数据集名 |
| `SETTING = "class-il"` | class-incremental learning 设置 |
| `N_CLASSES = 101` | Food101N/Food-101 总类别数 |
| `N_TASKS = 5` | 5 个增量任务 |
| `N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]` | 每个 task 包含多少新类别 |

Mammoth 的 `ContinualDataset.get_offsets()` 会根据 `N_CLASSES_PER_TASK` 算每个 task 的类别边界：

```text
task 0: start_c = 0,  end_c = 20
task 1: start_c = 20, end_c = 40
task 2: start_c = 40, end_c = 60
task 3: start_c = 60, end_c = 80
task 4: start_c = 80, end_c = 101
```

然后 `store_masked_loaders()` 用这个类别范围筛样本：

```python
train_mask = np.logical_and(train_dataset.targets >= start_c,
                            train_dataset.targets < end_c)
```

也就是说，当前 task 的 train loader 只包含当前类别范围内的训练样本。

如果 evaluation mode 是 `complete`，测试集会包含到当前 task 为止所有已经见过的类别：

```python
test_mask = np.logical_and(test_dataset.targets >= 0,
                           test_dataset.targets < end_c)
```

这就是 class-incremental evaluation：训练逐 task 进行，测试逐步扩展到所有已见类别，最后 Task 4 后评估全部 101 类。

## 数据处理代码具体参考了哪些论文

### 1. 项目原文 NRGP / DGC+CBP

论文：`Noise-Robust Class-Incremental Learning via Dynamic Gradient Constraint and Cumulative Bias Projection`

本代码参考它确定这些协议：

| 内容 | 采用到代码中的方式 |
|---|---|
| Food101N 是 real-world noisy dataset | `get_data_loaders()` 中拒绝 `--noise_rate > 0` |
| Food101N 不再注入 synthetic noise | 不走 Mammoth 通用 synthetic-noise pipeline |
| 101 类分 5 个 task | `N_CLASSES = 101`，`N_TASKS = 5` |
| 每 task 类别数 `[20,20,20,20,21]` | `N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]` |
| train 使用 noisy class labels | `self.targets` 在 train split 下就是 noisy targets |
| test 使用 clean labels | test split 复用 Food-101 clean metadata |

### 2. AER 论文

论文：`May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels`

本代码参考它确认 Seq. Food-101N 的复现实验背景：

| 内容 | 对本仓库的影响 |
|---|---|
| Seq. Food-101N 使用 5 tasks | 与原文一致，写入 `N_TASKS = 5` |
| Food-101N 训练图像为 224x224 输入规模 | transform 使用 `Resize(256)` + `RandomCrop(224)` |
| Food101N 实验使用 ResNet34、20 epochs、batch size 32 | 文档中标记当前默认值和论文设置差异 |
| Food101N 是真实噪声流 | 不注入额外 synthetic noise |

注意：AER 论文里的部分实现和 PuriDivER repo 有 GPL-3.0 许可证限制。本仓库没有复制其 Dataset 代码，只参考了论文公开协议和公开数据组织语义。

### 3. CleanNet / Food101N 论文和官方数据说明

论文：`CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise`

Food101N 来自 CleanNet 工作。本代码参考它确认：

| Food101N 概念 | 代码处理 |
|---|---|
| train labels 是 noisy class labels | train `self.targets` 直接用 metadata/class folder 给出的 label |
| verification labels 只是“该 noisy label 是否正确”的标记 | 代码保存为 `self.verification_labels`，但不替换 `self.targets` |
| Food101N classification evaluation 使用 Food-101 test set | test split 支持传入 Food-101 test metadata/images |
| 图片从 web 收集，路径结构可能不统一 | metadata/parser 支持多种字段名和目录结构 |

### 4. Food-101 论文

论文：`Food-101: Mining Discriminative Components with Random Forests`

本代码参考它确认 Food-101 的 clean test 语义：Food-101 是 101 类食物图像数据集，测试集每类有人工检查过的 clean labels。Food101N 的测试部分应使用 Food-101 clean test，而不是从 noisy train 里再拆测试集。

## 数据处理代码具体参考了哪些开源代码

### 1. Mammoth continual learning framework

参考位置：

```text
datasets/utils/continual_dataset.py
```

关键影响：

1. Dataset 必须有 `data` 和 `targets` 字段，才能被 `MammothDatasetWrapper` 包装。
2. 类增量任务通过 `store_masked_loaders()` 按 label 范围切分。
3. 非均匀 task split 可以用 list 类型的 `N_CLASSES_PER_TASK` 表达。

所以 Food101N 实现成：

```python
self.data = np.array(data, dtype=object)
self.targets = np.array(targets, dtype=np.int64)
```

这样才能让 Mammoth 在每个 task 自动筛出当前 task 的样本。

### 2. torchvision `Food101`

参考位置：

```text
torchvision.datasets.Food101
```

主要参考点：

1. Food-101 metadata 常见形式是 `meta/train.json`、`meta/test.json`。
2. JSON 可表现为 `{class_name: [image_id, ...]}`。
3. 类别名需要形成稳定的 `class_to_idx` 映射。

本仓库没有直接继承 `torchvision.datasets.Food101`，原因是 Food101N 的 train 部分不是标准 Food-101 train，它需要 noisy labels、verification labels 和更灵活的 metadata 字段。

### 3. PuriDivER 开源仓库

参考位置：

```text
https://github.com/clovaai/puridiver
```

参考点：

1. Food-101N 作为真实噪声 continual / online continual learning 数据集使用。
2. 数据可以整理成 `train/<class_name>/...`、`test/<class_name>/...` 这类 class-folder 结构。
3. Food-101N/WebVision 这类真实噪声数据通常不再人为加 CIFAR 式 symmetric/asymmetric noise。

没有复制它的 Dataset 实现，主要原因是：

1. 它的许可证是 GPL-3.0，不适合直接搬代码进当前项目。
2. 当前项目基于 Mammoth，需要接入 Mammoth 的 `ContinualDataset` 和 `store_masked_loaders()`。

### 4. CleanNet / Food101N 官方代码和页面

参考位置：

```text
https://github.com/kuanghuei/clean-net
https://kuanghuei.github.io/Food-101N/
```

主要参考数据字段语义，而不是复制代码：

1. Food101N 的 noisy label 是训练类别标签。
2. verification label 不是 clean class label。
3. Food101N 评估采用 Food-101 testing set。

### 5. cleanlab label-error-detection benchmarks

参考位置：

```text
https://github.com/cleanlab/label-error-detection-benchmarks
```

主要参考 Food101N Kaggle 解压后可能出现的 metadata/verification 文件线索，例如 `verified_train.tsv` 这类名字。当前实现因此支持 TSV/CSV/TXT/JSON 多种 metadata。

## 当前代码为什么这样处理 Food101N

### 1. 为什么不自动下载

Food101N 官方数据通常需要从 Kaggle 或原始发布源手动准备，Food-101 clean test 也可能在另一个目录里。本仓库不做自动下载，避免下载失败、路径不一致和数据许可证问题。

代码提供这些参数让用户显式指定路径：

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

### 2. 为什么支持多种 metadata

因为 Food101N 官方包、Food-101 官方包、PuriDivER/AER split、用户自己整理的数据，字段名可能不同。代码兼容这些字段：

| 类型 | 支持字段 |
|---|---|
| 图片路径 | `file_name`, `filepath`, `file_path`, `path`, `image`, `img`, `sample_key` |
| 类名 | `klass`, `class`, `class_name`, `category`, `category_name`, `label_name` |
| 类别 id | `label`, `target`, `class_idx`, `class_id`, `category_id` |
| verification | `verification_label`, `verified`, `is_verified` |

也支持：

```text
JSON list
JSON dict with annotations/samples/data/images/items
Food-101 style JSON dict
TXT/CSV/TSV
class folder scan
```

这样做是为了让同一份代码既能读 Food-101 官方 metadata，也能读 Food101N noisy train metadata，后续如果拿到 AER 作者 split，也可以直接通过 `--food101n_train_list` 接入。

### 3. 为什么 train label 用 noisy label

Food101N 的关键点就是训练标签带真实噪声。如果把 train label 改成 clean label，就不是噪声类增量学习了。

所以代码里：

```python
self.targets = np.array(targets, dtype=np.int64)
self.noisy_targets = self.targets.copy() if self.train else None
```

train split 的 `targets` 就是 noisy label。

### 4. 为什么 verification label 不参与替换

Food101N 的 verification label 通常只表示“这张图给定的 noisy class label 是否正确”，不是 101 类类别编号。

例如：

```text
image_path = pizza/xxx.jpg
noisy class label = pizza
verification label = 1 或 0
```

这里的 `1/0` 只是正确性标记，不是类别 id。如果把 verification label 当成 class label，会把 101 类分类任务错误变成二分类或错误标签任务。

因此代码只保存：

```python
self.verification_labels = np.array(verification_labels, dtype=np.int64)
```

不使用它覆盖 `self.targets`。

### 5. 为什么 test 复用 train 的 `class_to_idx`

Food101N noisy train 和 Food-101 clean test 可能来自不同目录或不同 metadata 文件。如果二者各自排序类别，可能出现 label id 错位。

所以 test dataset 创建时传入：

```python
class_to_idx=train_dataset.class_to_idx
```

这样 train/test 使用同一套 101 类映射。

## 一句话回答

这份 Food101N 数据处理代码的论文依据是项目原文 NRGP/DGC+CBP、AER、CleanNet/Food101N 和 Food-101；代码结构依据是 Mammoth 的 ContinualDat`aset/store_masked_loaders`，metadata 读取参考了 `torchvision.datasets.Food101` 和 Food101N/PuriDivER 公开数据组织方式，但没有直接复制 PuriDivER 或 CleanNet 的 Dataset 代码。

增量设置是 class-incremental：Food101N 的 101 类被切成 5 个任务 `[20, 20, 20, 20, 21]`，模型按 task 顺序学习新类别，旧 task 原始训练数据不可直接访问，只能通过有限 replay buffer 保留少量旧样本；最终评估所有已见类别，最后一个 task 后评估全部 101 类。
