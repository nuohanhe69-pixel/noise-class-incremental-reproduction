# Food101N Phase 1 Project Dataset Audit

日期：2026-07-14

## 范围

本阶段只调查当前 `mammoth_code` 的真实数据集、任务划分、训练接口和 buffer 调用链，不修改代码。代码根目录为：

`/Users/hunk/Documents/deep learning/噪声类增量项目复现/mammoth_code`

主要参考文件为 `AGENTS.md` 和 `readme_latest.md`。普通 `README.md` 只作为 Mammoth 原框架辅助资料。

## 当前项目结构

| 模块 | 文件 | 结论 |
|---|---|---|
| 入口 | `main.py`, `utils/main.py` | 解析参数、加载 dataset/model/backbone、进入训练 |
| 参数 | `utils/args.py` | `--dataset` 选择数据集；dataset 类 `__init__` 签名会变成动态参数 |
| 数据集注册 | `datasets/__init__.py` | 自动 import `datasets/*.py`；继承 `ContinualDataset` 且有 `NAME` 的类会被注册 |
| 数据集基类 | `datasets/utils/continual_dataset.py` | `store_masked_loaders()` 负责 task 过滤、noisy label 注入、DataLoader 创建 |
| 训练循环 | `utils/training.py` | batch 前 3 项必须是 `inputs, labels, not_aug_inputs`，额外字段按名称传给 `observe()` |
| 评估 | `utils/evaluate.py` | 通过 `dataset.get_offsets(k)` 做 task mask，Class-IL 使用当前已见类别范围 |
| buffer | `utils/buffer.py` | buffer 存 `examples, labels, true_labels` 等；`examples` 通常来自 `not_aug_inputs` |

## 数据集注册方式

`datasets/__init__.py` 的旧式注册会扫描 `datasets/*.py`，导入模块后寻找继承 `ContinualDataset` 的类，并使用类属性 `NAME` 作为命令行数据集名。`NAME='seq-food101n'` 可被 `--dataset seq-food101n` 选中，不需要额外改注册表。

如果数据集类 `__init__(self, args, food101n_root=None, ...)` 定义额外参数，`utils.args.add_dynamic_parsable_args()` 会把它们加入 CLI。

## Dataset 返回格式

普通 class-incremental 数据集应返回：

```text
(augmented_image, target, not_aug_image)
```

`MammothDatasetWrapper` 要求底层 dataset 至少有：

```text
data
targets
```

`store_masked_loaders()` 会根据 task mask 过滤 `data`、`targets`、`indexes`，并将当前 test loader append 到 `dataset.test_loaders`。

训练循环会读取 `train_loader.dataset.extra_return_fields`，把第 4 项及以后按字段名传给 `model.meta_observe()`。

## 类增量任务划分

`ContinualDataset.get_offsets(task_idx)` 已支持两种形式：

```text
N_CLASSES_PER_TASK = 20
N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]
```

对 list 的处理是前缀和：

```text
Task 0: [0, 20)
Task 1: [20, 40)
Task 2: [40, 60)
Task 3: [60, 80)
Task 4: [80, 101)
```

因此 Food101N 可以直接声明：

```text
N_CLASSES = 101
N_TASKS = 5
N_CLASSES_PER_TASK = [20, 20, 20, 20, 21]
```

已发现的非均匀 task 先例：

| 数据集 | 文件 | 任务大小 |
|---|---|---|
| MIT67 | `datasets/seq_mit67.py` | `[7] * 7 + [6] * 3` |
| Cars196 | `datasets/seq_cars196.py` | `[20] * 9 + [16]` |

## 兼容风险

| 区域 | 结论 |
|---|---|
| `store_masked_loaders()` | 正常 task split 支持 list offset |
| `utils/evaluate.py` | `mask_classes()` 和当前已见类别范围使用 `get_offsets()`，支持 list |
| `utils/buffer.py` | 多处显式判断 int/list，支持 list |
| `_get_mask_unlabeled()` | 当 `label_perc != 1` 时使用 `train_dataset.targets.shape[0] // setting.N_CLASSES_PER_TASK`，list 会不兼容；Food101N 本阶段不启用半监督百分比 |
| `models/utils/continual_model.py` | `_cpt = dataset.N_CLASSES_PER_TASK` 可能变成 list；常规 OGC/SAP/AER 路径主要用 `get_offsets()`，但不是所有历史模型都保证支持 list |

## 当前 Food101N 文件现状

本地已有未跟踪文件 `datasets/seq_food101n.py`，但当前实现不满足本任务要求：

1. `N_CLASSES_PER_TASK = 20` 会遗漏第 101 类或错误处理最后任务。
2. `get_data_loaders()` 使用 `self.i`，而当前 `ContinualDataset` 初始化的是 `self.c_task`，没有 `self.i`。
3. 直接创建 `DataLoader`，绕过 `store_masked_loaders()`，因此不会维护 `dataset.test_loaders`，训练器断言也可能失败。
4. 依赖 PuriDivER JSON 文件名，但没有说明来源、许可证和字段语义。
5. 训练 target 优先使用 `true_label`，这会把真实 noisy train protocol 改成 clean train protocol。
6. 没有 `.data`/`.targets` 兼容 `MammothDatasetWrapper`。
7. 均值方差写成 `(0,0,0)/(1,1,1)`，没有说明与 Food101N/ResNet 输入的依据。

结论：应重写为 Food101N raw dataset + `SequentialFood101N` continual wrapper，并只保留必要兼容思路。

## readme_latest 相关结论

`readme_latest.md` 当前只给出 CIFAR10/CIFAR100/NTU60 复现命令，没有 Food101N 命令。主方法包括：

```text
er-ace-aer-abs
aer-sap
ogc-sap --enable_sap 0
ogc-sap --enable_sap 1
```

本阶段不修改 `readme_latest.md`，只在最终汇报建议后续同步 Food101N 命令。

## 最小实现边界

本阶段应只修改或新增：

1. `datasets/seq_food101n.py`
2. `datasets/configs/seq-food101n/default.yaml`
3. `docs/food101n/*.md`
4. 按 `AGENTS.md` 要求更新 `AGENTS.md`

不应修改算法、loss、optimizer、buffer 策略、评估统计、CIFAR 默认配置或正式实验命令。
