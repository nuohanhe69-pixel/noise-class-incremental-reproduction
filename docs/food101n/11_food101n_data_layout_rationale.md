# 为什么 Food101N 数据目录和 CIFAR/NTU60 不一样

日期：2026-07-14

## 结论

`data/Food-101N/meta/train.tsv`、`data/Food-101N/meta/test.tsv`、`data/Food-101N/meta/classes.txt`、`data/Food-101N/images/` 不是 Food101N 官方唯一固定格式，而是本项目为 Food101N 复现设计的推荐标准化格式。

它和 CIFAR10/100、NTU60 不一样，是因为三类数据本身完全不同：

| 数据集 | 数据形态 | 本项目读取方式 | 为什么可以这样 |
|---|---|---|---|
| CIFAR10/100 | 小尺寸图像，torchvision 已有标准 Dataset | 继承 `torchvision.datasets.CIFAR10/CIFAR100`，自动下载/校验 | torchvision 已经封装好 train/test、labels、classes |
| NTU60 | 骨架动作序列，不是普通图片 | 读一个 `NTU60_CS.npz`，里面有 `x_train/y_train/x_test/y_test` | 预处理后数组更适合骨架模型 EfficientGCN |
| Food101N | 大规模真实 web 图片 + noisy labels + clean test 来源不同 | jpg 图片放在 `images/`，split 和 label 放在 `meta/*.tsv` | 必须显式限定 `52,867/4,741` split，不能靠扫描目录猜 |

所以 Food101N 目录长这样，不是为了“和 CIFAR/NTU 故意不一样”，而是为了把真实噪声图像数据、类别顺序、train/test split、noisy/clean label 语义都固定下来。

## CIFAR10/100 为什么不用这种目录

CIFAR10/100 的代码在本项目里使用 torchvision：

```python
from torchvision.datasets import CIFAR10

train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True, download=True)
test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False, download=True)
```

torchvision 已经知道 CIFAR 的标准结构：

1. 哪些文件是 train。
2. 哪些文件是 test。
3. 每张图片的 label 是什么。
4. 10/100 个 class name 是什么。
5. 数据完整性怎么检查。

所以本项目不需要额外写：

```text
data/CIFAR10/meta/train.tsv
data/CIFAR10/meta/test.tsv
data/CIFAR10/images/
```

CIFAR 数据本身也很小，通常会被 torchvision 解包成内部 batch 文件或 Python pickle，而不是一张一张 jpg 配一个 metadata 表。

## NTU60 为什么是 `.npz`

NTU60 不是图像分类数据，而是骨架动作识别数据。它的每个样本不是一张 jpg，而是一段人体骨架序列。

本项目代码要求：

```text
data/NTU60_CS.npz
```

并且 `.npz` 里必须有：

```text
x_train
y_train
x_test
y_test
```

原因是 NTU60 的模型输入是类似这样的张量：

```text
[N, C, T, V, M]
```

其中：

| 符号 | 含义 |
|---|---|
| `N` | 样本数 |
| `C` | 坐标通道数，一般是 3 |
| `T` | 时间帧数 |
| `V` | 关节点数 |
| `M` | 人数/骨架数 |

因此它天然更适合预处理成一个 `.npz` 数组包，而不是 `images/` 目录。

## Food101N 为什么需要 `meta/*.tsv`

Food101N 比 CIFAR/NTU 更麻烦，原因有 5 个。

### 1. Food101N 是真实 noisy label 数据集

Food101N train label 本身就是 noisy class label。它不像 CIFAR 那样从 clean labels 出发再人工注入 `symm/asym` 噪声。

因此必须明确记录每张训练图对应的 noisy class label。

### 2. Food101N 全量数据和论文 split 不一样

Food101N 官方全量大约有 310k 张 web images。

但当前项目已经确认采用论文/AER/NTD 接近的：

```text
52,867 train / 4,741 test
```

如果代码只是扫描：

```text
data/Food-101N/images/<class_name>/*.jpg
```

它很可能会扫到完整 310k 数据，或者扫到一个不确定的子集。这样训练结果就不能和原文/AER/NTD 表格对齐。

所以必须用 `train.tsv` 和 `test.tsv` 显式指定“到底哪些图片属于本次复现 split”。

### 3. Food101N train 和 test 来源可能不是同一个包

Food101N 的 train 是 noisy web images。

Food101N/Food-101 的 test 通常来自 clean Food-101 test set，或者来自 AER/NTD 已整理好的 test split。

也就是说，train/test 可能来自两个不同目录：

```text
Food101N noisy train
Food-101 clean test
```

为了让训练入口稳定，推荐把它们整理成统一项目格式：

```text
data/Food-101N/
  images/
  meta/
```

如果不想移动图片，代码也支持分开指定：

```bash
--train-images-dir train
--test-images-dir ../food-101/images
```

### 4. verification label 不是 clean class label

Food101N 里可能有 verification label，但这个值不是 101 类类别编号。

它通常只表示：

```text
这张图给定的 noisy class label 是否被人工判断为正确
```

所以不能把 verification label 当成训练标签。`train.tsv` 里可以保存它，但代码只把它当辅助字段，不用它覆盖 `targets`。

### 5. class order 必须固定

类增量学习按类别编号切 task：

```text
Task 0: labels [0, 20)
Task 1: labels [20, 40)
Task 2: labels [40, 60)
Task 3: labels [60, 80)
Task 4: labels [80, 101)
```

如果 class order 不固定，今天 label 0 是 `apple_pie`，明天 label 0 变成 `baby_back_ribs`，那 task split 就完全变了。

所以需要：

```text
meta/classes.txt
```

来固定 101 个类别的顺序。

## 推荐目录每个文件是什么

推荐格式：

```text
data/Food-101N/
  meta/
    train.tsv
    test.tsv
    classes.txt
  images/
    apple_pie/
      xxx.jpg
    baby_back_ribs/
      yyy.jpg
    ...
```

### `images/`

保存真实图片文件。路径一般按类别分文件夹：

```text
images/<class_name>/<image_name>.jpg
```

也可以不完全按这个结构，只要 `train.tsv/test.tsv` 里的 `file_name` 能正确指向图片。

### `meta/train.tsv`

保存 Food101N noisy train split。

推荐带 header：

```tsv
file_name	klass	label	verification_label
apple_pie/000001.jpg	apple_pie	0	1
apple_pie/000002.jpg	apple_pie	0	0
baby_back_ribs/000003.jpg	baby_back_ribs	1	-1
```

字段含义：

| 字段 | 含义 | 是否必须 |
|---|---|---|
| `file_name` | 相对 `images/` 的图片路径 | 必须 |
| `klass` | 类别名 | 建议提供 |
| `label` | 0 到 100 的类别 id | 建议提供 |
| `verification_label` | noisy label 是否被验证正确；没有就填 `-1` 或不填 | 可选 |

如果同时提供 `klass` 和 `label`，代码会检查它们是否和 `classes.txt` 一致。

### `meta/test.tsv`

保存 clean test split。

推荐格式类似：

```tsv
file_name	klass	label
apple_pie/100001.jpg	apple_pie	0
baby_back_ribs/100002.jpg	baby_back_ribs	1
```

test 的 label 应该是 clean class label。

### `meta/classes.txt`

保存 101 类类别顺序，一行一个类名：

```text
apple_pie
baby_back_ribs
baklava
...
```

这个文件决定：

```text
第 1 行 -> label 0
第 2 行 -> label 1
第 3 行 -> label 2
...
```

如果写成：

```text
0 apple_pie
1 baby_back_ribs
2 baklava
```

当前代码也能读。

## 为什么不直接照官方原始目录

官方原始目录通常不是为了 class-incremental reproduction 设计的。

我们当前要解决的是：

1. 必须固定 `52,867 / 4,741` split。
2. 必须固定 101 类 class order。
3. 必须区分 noisy train label 和 clean test label。
4. 必须兼容 Mammoth 的 `data/targets` 接口。
5. 必须避免不小心扫入完整 310k 数据。

因此标准化成 `meta + images` 是为了让复现实验可控、可检查、可复用。

## 这是不是唯一能用的结构

不是。

当前代码也支持其他路径形式，比如：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N_release \
  --train-list meta/train.tsv \
  --test-list ../food-101/meta/test.txt \
  --train-images-dir train \
  --test-images-dir ../food-101/images \
  --classes-file ../food-101/meta/classes.txt \
  --strict
```

也就是说，如果你的真实数据已经分散在不同目录里，可以不移动图片，只要用参数告诉代码：

1. train metadata 在哪里。
2. test metadata 在哪里。
3. train 图片在哪里。
4. test 图片在哪里。
5. classes 文件在哪里。

## 一句话回答

CIFAR10/100 和 NTU60 的数据格式已经被 torchvision 或 `.npz` 预处理方式封装好了；Food101N 是大规模真实噪声 jpg 图片，而且当前复现必须精确限定 `52,867 train / 4,741 test` split，所以需要额外的 `meta/train.tsv`、`meta/test.tsv`、`meta/classes.txt` 来固定样本列表、标签和类别顺序。这个目录是本项目的推荐标准化格式，不是 Food101N 官方唯一格式。
