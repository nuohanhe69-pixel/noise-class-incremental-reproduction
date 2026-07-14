# Food-101N 57,608 张图片的来源与抽取方式

## 结论

当前项目中的 Food-101N 正式复现实验不是从完整 Food-101N 的约 310,009 张图片中重新随机抽样，而是采用 NTD/PuriDivER 已发布的 Food-101N 类增量任务列表。

也就是说，57,608 张图片来自一组已经写好的任务 JSON 文件：

- train: 5 个 `Food-101N_train_blurry10_rand1_cls20_task*.json`
- test: 5 个 `Food-101N_test_rand1_cls20_task*.json`

这些 JSON 文件中的每一行都指定了一张图片的相对路径、类别名、类别编号和 verification label。项目只把这些列表中出现的图片作为本次复现实验使用的数据。

## 数量怎么得到

本项目使用的是 `rand1 / cls20 / blurry10` 设置。

train 部分：

| task | JSON 文件 | 样本数 |
|---|---|---:|
| 0 | `Food-101N_train_blurry10_rand1_cls20_task0.json` | 10,436 |
| 1 | `Food-101N_train_blurry10_rand1_cls20_task1.json` | 10,424 |
| 2 | `Food-101N_train_blurry10_rand1_cls20_task2.json` | 10,425 |
| 3 | `Food-101N_train_blurry10_rand1_cls20_task3.json` | 10,461 |
| 4 | `Food-101N_train_blurry10_rand1_cls20_task4.json` | 11,121 |
| 合计 |  | 52,867 |

test 部分：

| task | JSON 文件 | 样本数 |
|---|---|---:|
| 0 | `Food-101N_test_rand1_cls20_task0.json` | 946 |
| 1 | `Food-101N_test_rand1_cls20_task1.json` | 921 |
| 2 | `Food-101N_test_rand1_cls20_task2.json` | 969 |
| 3 | `Food-101N_test_rand1_cls20_task3.json` | 933 |
| 4 | `Food-101N_test_rand1_cls20_task4.json` | 972 |
| 合计 |  | 4,741 |

因此总数是：

```text
52,867 train + 4,741 test = 57,608 images
```

## 每条记录长什么样

NTD/PuriDivER 的 task JSON 每条记录大致如下：

```json
{
  "file_name": "caprese_salad/669be1415f3363cce83bd0b1873c6a1a.jpg",
  "verification_label": 1,
  "klass": "caprese_salad",
  "label": 77
}
```

字段含义：

- `file_name`: 图片在 Food-101N 图片根目录下的相对路径。
- `klass`: 类别名。
- `label`: 类别编号。
- `verification_label`: Food-101N 的人工验证标记，`1` 通常表示该图片和给定类别一致，`0` 表示不一致或可疑。它不是新的类别标签，而是噪声监督信息。

## “抽取”实际发生在哪里

有两层含义：

第一层是实验协议层面：NTD/PuriDivER 已经发布了这些 task JSON。它们决定了哪些图片进入 train，哪些图片进入 test，以及每个增量阶段使用哪些类别。我们没有重新随机生成 57,608 张图片列表，而是继承这个公开 split。

第二层是项目落地层面：本项目的 `scripts/prepare_food101n_metadata.py` 读取这些 task JSON，然后生成项目统一使用的：

```text
data/Food-101N/meta/train.tsv
data/Food-101N/meta/test.tsv
data/Food-101N/meta/classes.txt
```

脚本逻辑是：

1. 按 `task0` 到 `task4` 读取 5 个 train JSON。
2. 按 `task0` 到 `task4` 读取 5 个 test JSON。
3. 分别合并为 train/test 两个列表。
4. 检查 train 数量是否为 52,867，test 数量是否为 4,741。
5. 统一写成 Mammoth 项目更方便读取的 TSV 元数据。
6. 如果之后提供了真实 Food-101N 图片目录，并启用 `--copy-images` 或 `--link-images`，脚本才会把这些列表中出现的图片复制或软链接到项目数据目录。

所以目前已经准备好的主要是“57,608 张图片的清单”。真正的图片文件仍然需要从 Food-101N 数据源下载后，按这些相对路径匹配。

## 增量设置是什么

采用 NTD/PuriDivER 的 Food-101N 设置：

- 总类别数：101
- 任务数：5
- 每个 task 文件名中使用 `cls20`
- 模型设置参考 NTD：ResNet34
- memory size：2,000
- 使用 `rand1` 作为当前项目的正式复现 split

需要注意，Food-101N 有 101 类，无法被 5 个任务完全均分。因此 `cls20` 是文件命名和任务设置中的类别步长标识，最后一个任务会承接剩余类别，样本数也会更大。

## 为什么不是完整 310,009 张

完整 Food-101N 约有 310,009 张图片，但论文复现协议采用的是更接近 NTD/PuriDivER 的已发布 Food-101N 增量任务 split，而不是把完整数据集全部用于训练。

这样做的原因是：

- 和 NTD/PuriDivER 的 Food-101N 实验设置对齐。
- 可以使用公开 task JSON 精确复现 train/test 列表。
- 避免自己重新划分导致结果不可比较。
- 保留 `verification_label`，便于噪声鲁棒类增量方法使用。

## 当前项目对应文件

- 任务列表来源：`external/ntd/tasks/Food-101N/`
- 元数据生成脚本：`scripts/prepare_food101n_metadata.py`
- 本项目生成后的 metadata：
  - `data/Food-101N/meta/train.tsv`
  - `data/Food-101N/meta/test.tsv`
  - `data/Food-101N/meta/classes.txt`

## 重要说明

NTD 仓库公开了最终 task JSON，并且训练代码会直接读取这些 JSON；但在当前可见源码中，没有看到 Food-101N 这些 JSON 最初如何从完整 Food-101N 中生成的脚本。因此严谨说法是：

本项目复现的是 NTD/PuriDivER 发布的 Food-101N 增量任务列表，而不是自行复刻其原始抽样过程。

