# Food101N Phase 5 Validation Report

日期：2026-07-14

## 执行环境

当前工作目录：

`/Users/hunk/Documents/deep learning/噪声类增量项目复现/mammoth_code`

当前系统 Python：

```text
python3 = macOS system Python 3.9.6
```

当前解释器缺少：

```text
torch
torchvision
yaml
```

因此无法在本环境完成真实 import、dataset 注册、DataLoader batch 或 `main.py --help` 验证。

## 执行的命令

```bash
python3 -m py_compile datasets/seq_food101n.py
```

结果：通过。

```bash
python3 - <<'PY'
try:
    import datasets.seq_food101n as food
    print('import ok', food.SequentialFood101N.NAME)
except Exception as e:
    print(type(e).__name__ + ':', e)
PY
```

结果：

```text
ModuleNotFoundError: No module named 'torch'
```

```bash
python3 - <<'PY'
from pathlib import Path
try:
    import yaml
except Exception as e:
    print('yaml import failed:', repr(e))
else:
    data = yaml.safe_load(Path('datasets/configs/seq-food101n/default.yaml').read_text())
    print(data)
PY
```

结果：

```text
yaml import failed: ModuleNotFoundError("No module named 'yaml'")
```

```bash
python3 - <<'PY'
import ast
from pathlib import Path
module = ast.parse(Path('datasets/seq_food101n.py').read_text())
for n in module.body:
    if isinstance(n, ast.ClassDef) and n.name == 'SequentialFood101N':
        for key in ('NAME', 'N_CLASSES', 'N_TASKS', 'N_CLASSES_PER_TASK'):
            for stmt in n.body:
                if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in stmt.targets):
                    print(f'{key}: {ast.literal_eval(stmt.value)}')
PY
```

结果：

```text
NAME: seq-food101n
N_CLASSES: 101
N_TASKS: 5
N_CLASSES_PER_TASK: [20, 20, 20, 20, 21]
```

## 静态验证结果

| 检查项 | 结果 |
|---|---|
| Python 语法检查 | 通过 |
| `Food101N` raw dataset 类存在 | 通过 |
| `SequentialFood101N` continual wrapper 存在 | 通过 |
| 注册名 `seq-food101n` | 通过静态检查 |
| 总类别数 101 | 通过静态检查 |
| 任务数 5 | 通过静态检查 |
| 每任务类别数 `[20, 20, 20, 20, 21]` | 通过静态检查 |
| 是否调用 `store_masked_loaders()` | 通过静态检查 |
| 是否拒绝 `--noise_rate > 0` | 通过代码检查 |
| 是否修改算法代码 | 未修改 |
| 是否修改 `readme_latest.md` | 未修改 |

## 数据完整性验证

本机未发现真实 Food101N 数据：

```text
未发现 data/Food-101N
未发现 data/Food-101N_release
未发现 Food-101 clean test 目录
```

因此以下验证未执行：

1. train metadata 读取。
2. test metadata 读取。
3. 训练样本数量。
4. 测试样本数量。
5. 缺失图像检查。
6. train/test 路径重叠检查。
7. 图像 RGB 读取。
8. 单样本返回。
9. 单 batch 返回。

未执行原因：缺真实数据，且当前 Python 环境缺 `torch`/`torchvision`。

## 任务划分验证

静态 task split：

```text
Task 0 classes: 20
Task 1 classes: 20
Task 2 classes: 20
Task 3 classes: 20
Task 4 classes: 21
Union classes: 101
Intersection between tasks: empty by construction
Missing classes: none by construction
Duplicate classes: none by construction
Boundaries: [0, 20, 40, 60, 80, 101]
```

当前框架 `ContinualDataset.get_offsets()` 已支持 list task size；Food101N 使用该机制，不修改通用训练器。

## DataLoader 验证

未执行。原因：

1. 当前 Python 缺 `torch` 和 `torchvision`。
2. 本机没有真实 Food101N/Food-101 数据。

在可运行环境和数据准备完成后，应验证：

```text
image shape: [B, 3, 224, 224]
not_aug image shape: [B, 3, 224, 224]
label shape: [B]
label range: 0..100
train labels: noisy class labels
test labels: clean Food-101 labels
```

## 当前已知问题

1. Food101N 官方没有公开 class-incremental class order；当前实现使用稳定类别顺序，或由用户通过 metadata/class order 指定。
2. Food101N train 无完整 clean class label；不能把 verification label 当 clean class label。
3. `er-ace-aer-abs.observe()` 当前强制要求 `true_labels`，真实 Food101N 协议下需要后续确认处理方式。
4. 当前环境不能完成 import/注册测试。
5. 当前未验证 Kaggle 解压后的完整文件名和官方样本数。

## 是否可以进入下一阶段

结论：Dataset 已完成，但缺少真实数据验证和可运行 Python 环境验证。需要用户准备 Food101N/Food-101 数据并切换到安装 `torch`、`torchvision`、`pyyaml` 的项目环境后，才能进入训练接入阶段。
