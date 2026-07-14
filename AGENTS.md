# AGENTS.md

## 1. 文件用途与适用范围

本文件是 `mammoth_code` 代码仓库的项目级 AI Agent 操作说明，供 Codex、Claude Code、其他 AI 编程代理以及后续开发者使用。它位于仓库根目录，对整个 `mammoth_code` 生效。

本文件不是论文理论笔记，也不是实验结果汇总；它记录当前代码、配置、脚本、数据、checkpoint、输出和长期维护规则。任何 Agent 修改代码、配置、实验脚本、数据流程或复现实验协议前，应先阅读本文件，再阅读 `readme_latest.md` 和相关代码。

未来如果某个子目录需要特殊规则，可以在该子目录新增局部 `AGENTS.md`。局部规则只能补充或细化本文件，不应在没有明确说明时与根目录规则冲突。

## 2. 项目根目录与主要参考文档

代码仓库根目录是当前目录 `mammoth_code/`。外层 `噪声类增量项目复现/` 是资料总目录，包含实验报告、压缩包、数据备份和辅助材料，不是本次代码仓库主体。

主要项目说明文件是 `readme_latest.md`。普通 `README.md` 只作为 Mammoth 原框架和历史说明的辅助参考，不能优先于当前代码和 `readme_latest.md`。

参考优先级：

1. 当前实际代码和真实调用关系。
2. `readme_latest.md`。
3. 当前实际使用的配置文件、脚本、日志和结果记录。
4. 当前有效依赖文件。
5. 外层实验结果 Markdown 文档。
6. 普通 `README.md`、`REPRODUCIBILITY.md` 和原 Mammoth 文档。

已确认状态：

| 文件 | 作用 | 当前判断 |
|---|---|---|
| `readme_latest.md` | 当前噪声类增量实验推荐命令 | CIFAR10/CIFAR100 主命令基本能与当前代码对上；NTU60 示例存在模型名差异，见待确认事项 |
| `README.md` | 原 Mammoth README 加部分历史实验片段 | 包含旧模型名和不完整/不一致命令，只能辅助参考 |
| `REPRODUCIBILITY.md` | 原 Mammoth 方法复现列表 | 属于上游框架参考，不代表本分支噪声类增量实验已复现 |

不得把外层 `.tar.gz` 压缩包、外层 Markdown 结果、`.pyc` 缓存或 `other/` 里的历史文件当成当前运行代码。

## 3. 项目概述

本项目是基于 Mammoth 持续学习框架的二次修改分支，研究方向是噪声标签场景下的类增量学习。当前核心任务是 sequential CIFAR-10 / CIFAR-100 等 class-incremental learning 设置，在训练标签存在 symmetric 或 asymmetric noise 时复现或改进噪声鲁棒类增量方法。

当前代码中的主要 baseline 和改进方法：

| 名称 | 注册模型名 | 文件 | 当前作用 |
|---|---|---|---|
| AER/ABS baseline | `er-ace-aer-abs` | `models/er_ace_aer_abs.py` | 基于 ER-ACE、AER 和 ABS 的噪声鲁棒 rehearsal baseline |
| SAP only | `aer-sap` | `models/aer_sap.py` | 在 AER/ABS 上加入 SAP，用于消融 |
| OGC + optional SAP | `ogc-sap` | `models/ogc_sap.py` | 当前 `readme_latest.md` 推荐主方法 |
| ER / ER-ACE | `er`, `er-ace` | `models/er.py`, `models/er_ace.py` | Mammoth 原有/基础 replay baseline，可辅助比较 |

当前主要数据集：

| 数据集 | 注册名 | 文件 | 状态 |
|---|---|---|---|
| CIFAR-10 | `seq-cifar10` | `datasets/seq_cifar10.py` | 代码支持；本仓库 `data/CIFAR10` 已存在 |
| CIFAR-100 | `seq-cifar100` | `datasets/seq_cifar100.py` | 代码支持；当前未发现 `data/CIFAR100` |
| NTU60 | `seq-ntu60` | `datasets/seq_ntu60.py` | 代码支持注册；当前未发现 `data/NTU60_CS.npz`，运行命令待验证 |
| Food-101N | `seq-food101n` | `datasets/seq_food101n.py` | 数据集接入已实现；当前未发现真实 Food-101N/Food-101 数据，尚未训练验证 |

## 4. 复现目标与实验范围

本项目应区分以下实验类型，不得混写：

| 类型 | 定义 | 当前证据 |
|---|---|---|
| 原论文严格复现 | 数据、class order、noise labels、seed、backbone、epoch、batch size、buffer、optimizer、指标均与论文一致 | 待确认 |
| 资源受限复现 | 因设备或环境调整 batch size、worker、设备或部分超参 | 已有本机 MPS 运行记录，见 `cifar10_symm40_readme_latest_local_retry2_20260703_231106_analysis.md` |
| 参数调整实验 | 为追指标调整 OGC/SAP 超参 | 已有多份 CIFAR10 调参汇总 |
| baseline 实验 | `er-ace-aer-abs` 等 baseline 单独运行 | 部分记录存在，需按日志逐项确认 |
| 完整方法实验 | `ogc-sap --enable_sap 1` | CIFAR10 多个结果目录与日志存在 |
| 消融实验 | `er-ace-aer-abs`、`aer-sap`、`ogc-sap --enable_sap 0/1` 对比 | `readme_latest.md` 给出命令；完整执行状态待确认 |

当前已开展的证据主要集中在 CIFAR10；CIFAR100 有 checkpoint 和外层整理文档；NTU60 目前更多是代码和结果文档线索，数据文件不在当前仓库。

## 5. 当前项目状态

以下状态基于 2026-07-14 的只读检查。

| 项目 | 状态 | 证据 |
|---|---|---|
| Git 仓库 | 已确认 | 当前目录是 Git 根目录，分支 `master`，落后 `origin/master` 3 个提交 |
| 工作树 | 存在大量既有改动 | 多个模型/测试文件删除或未跟踪，`readme_latest.md`、`models/ogc_sap.py` 等为未跟踪或修改状态 |
| 当前 shell 环境 | 未建立可运行训练环境 | `python` 不存在；`python3` 为 3.9.6，缺少 `torch`、`torchvision`、`sklearn` |
| 依赖声明 | 已存在 | `requirements.txt`、`requirements-optional.txt`、`pyproject.toml` |
| CIFAR-10 数据 | 已准备 | `data/CIFAR10/cifar-10-batches-py` 存在 |
| CIFAR-100 数据 | 未发现 | 未发现 `data/CIFAR100` |
| NTU60 数据 | 未发现 | 未发现 `data/NTU60_CS.npz` |
| Food-101N 数据 | 未发现 | 未发现 `data/Food-101N`、`data/Food-101N_release` 或 Food-101 clean test |
| CIFAR10 symmetric noisy label cache | 已存在部分 | `data/noisy_labels/seq-cifar10/symmetric/20/0/noisy_targets` 和 `40/0/noisy_targets` |
| CIFAR10 训练日志 | 已存在 | `run_logs/` 下多份 CIFAR10 日志 |
| checkpoint | 已存在 | `checkpoints/` 下有 CIFAR10 与 CIFAR100 `.pt` |
| 独立评估 | 代码支持，但当前环境未验证 | `--loadcheck`、`--inference_only`、`--start_from`、`--stop_after` 存在 |
| 多 seed | 进行中/待确认 | `readme_latest.md` 建议 seed 0 或 152，结果不完整 |
| NTU60 | 待验证 | 数据缺失，`readme_latest.md` 示例模型名未在 `models/` 注册 |

## 6. 仓库目录结构

精简结构：

```text
mammoth_code/
├── AGENTS.md
├── readme_latest.md
├── README.md
├── main.py                      # 当前主入口
├── requirements.txt
├── requirements-optional.txt
├── pyproject.toml
├── backbone/                    # backbone 注册与实现
│   ├── ResNetBlock.py            # resnet18 等 CIFAR backbone
│   └── EfficientGCN.py           # seq-ntu60 默认 backbone
├── datasets/
│   ├── __init__.py               # dataset 动态注册
│   ├── seq_cifar10.py
│   ├── seq_cifar100.py
│   ├── seq_ntu60.py
│   ├── configs/
│   └── utils/
│       ├── continual_dataset.py  # task 划分、noisy label 注入、DataLoader
│       └── label_noise.py        # symmetric/asymmetric noise
├── models/
│   ├── __init__.py               # model 动态注册
│   ├── er_ace_aer_abs.py
│   ├── aer_sap.py
│   ├── ogc_sap.py
│   ├── ogc_sap_old.py            # 疑似历史版本，尚未完全验证
│   └── config/
├── utils/
│   ├── args.py                   # argparse 参数定义
│   ├── main.py                   # 与 main.py 内容相同/类型变化，谨慎使用
│   ├── training.py               # train loop 与 evaluation/checkpoint 调用
│   ├── evaluate.py
│   ├── buffer.py
│   ├── checkpoints.py
│   ├── loggers.py
│   ├── schedulers.py
│   ├── sap_core.py
│   └── sap_model_utils.py
├── scripts/
│   └── prepare_ntu60_npz.py
├── data/                         # 数据、noisy label cache、结果；不要随意修改
├── checkpoints/                  # 权重；不要覆盖或删除
├── output/                       # 图、阈值、消融辅助输出
├── run_logs/                     # 运行日志
├── docs/                         # Mammoth 原文档
├── examples/                     # Mammoth notebook 示例
└── other/                        # 历史/辅助代码，尚未完全验证
```

## 7. 核心文件与运行状态

| 路径 | 作用 | 是否参与当前主流程 | 被谁调用 | 修改风险 | 验证状态 | 备注 |
|---|---|---:|---|---|---|---|
| `main.py` | 训练/评估统一入口 | 是 | 用户命令 | 高 | 代码阅读已确认 | 当前 shell 无法执行 |
| `utils/main.py` | 与 `main.py` 内容相同的入口副本 | 待确认 | README 提到 | 中 | Git 类型变化 | 优先用根目录 `main.py` |
| `utils/args.py` | 参数定义 | 是 | `main.py` | 高 | 已阅读 | 定义 noise、checkpoint、results、debug 等参数 |
| `utils/training.py` | 任务循环、训练、评估、保存 | 是 | `main.main()` | 高 | 已阅读 | `--inference_only` 仍会逐任务评估 |
| `utils/evaluate.py` | Class-IL/Task-IL 评估 | 是 | `dataset.evaluate()` | 高 | 已阅读 | 测试标签保持干净 |
| `datasets/utils/continual_dataset.py` | task 划分、noisy label 注入、DataLoader | 是 | 各 dataset | 高 | 已阅读 | 训练集 targets 会被 noisy targets 覆盖 |
| `datasets/utils/label_noise.py` | 噪声生成与缓存 | 是 | `store_masked_loaders()` | 高 | 已阅读 | asymmetric 只支持 CIFAR10/CIFAR100 |
| `datasets/seq_cifar10.py` | CIFAR10 类增量数据集 | 是 | `datasets.get_dataset()` | 高 | 已阅读 | 5 tasks，每任务 2 类 |
| `datasets/seq_cifar100.py` | CIFAR100 类增量数据集 | 是 | `datasets.get_dataset()` | 高 | 已阅读 | 默认 10 tasks，每任务 10 类 |
| `datasets/seq_ntu60.py` | NTU60 类增量数据集 | 待验证 | 注册为 `seq-ntu60` | 高 | 已阅读 | 数据文件缺失 |
| `models/er_ace_aer_abs.py` | AER/ABS baseline | 是 | `models.get_model()` | 高 | 已阅读 | 使用 `true_labels` 额外字段 |
| `models/aer_sap.py` | SAP-only 消融 | 是 | `models.get_model()` | 高 | 已阅读 | 依赖 SAP activation projection |
| `models/ogc_sap.py` | 当前主改进方法 | 是 | `models.get_model()` | 高 | 已阅读 | OGC + SAP，parser 定义关键超参 |
| `utils/buffer.py` | replay buffer 和采样策略 | 是 | rehearsal models | 高 | 已阅读 | 支持 reservoir/lars/labrs/abs/balancoir |
| `utils/checkpoints.py` | checkpoint 保存/加载 | 是 | `main.py`, `training.py` | 高 | 已阅读 | safe checkpoint 保存 args/results/buffer |
| `readme_latest.md` | 当前主要运行说明 | 是，文档参考 | 人/Agent | 中 | 已阅读 | 不得直接覆盖 |
| `README.md` | 上游 Mammoth + 历史片段 | 辅助 | 人/Agent | 中 | 已阅读 | 有旧命令与当前代码差异 |
| `checkpoints/1.py` | 硬编码权重转换脚本 | 否/辅助 | 手工运行 | 高 | 已阅读 | 不要直接执行或改源/目标路径 |
| `other/aer_ogc_sap.py` | 历史/辅助模型文件 | 否，未在 `models/` 注册 | 未确认 | 中 | 搜索确认 | 疑似历史遗留，尚未完全验证 |
| `models/ogc_sap_old.py` | OGC/SAP 旧版本 | 否/历史 | 未确认 | 中 | 部分搜索 | 疑似历史遗留，尚未完全验证 |

## 8. 代码调用链

当前真实主调用链：

```text
python main.py ...
→ main.parse_args()
→ utils.args.add_initial_args() 解析 dataset/model/backbone/loadcheck
→ models.get_model_class(args).get_parser() 加载模型专属参数
→ models.utils.load_model_config() 读取 models/config/<model>.yaml（若存在）
→ datasets.get_dataset_class() / datasets.utils.load_dataset_config()
→ utils.args.add_dynamic_parsable_args() 加载 dataset/backbone 动态参数
→ utils.args.add_management_args() / add_experiment_args()
→ update_cli_defaults(config)
→ parser.parse_args(cmd)，命令行覆盖配置默认值
→ set_random_seed(args.seed)
→ initialize()
→ utils.conf.get_device()
→ utils.conf.base_path(args.base_path)
→ utils.conf.get_checkpoint_path(args.checkpoint_path)
→ datasets.get_dataset(args)
→ extend_args(args, dataset)
→ backbone.get_backbone(args)
→ dataset.get_loss()
→ models.get_model(args, backbone, loss, dataset.get_transform(), dataset)
→ utils.training.train(model, dataset, args)
```

训练期调用链：

```text
train()
→ dataset.get_data_loaders()
→ store_masked_loaders()
→ 可选 build_noisy_labels()
→ create_seeded_dataloader()
→ model.meta_begin_task()
→ 每 epoch: model.meta_begin_epoch()
→ train_single_epoch()
→ model.meta_observe(inputs, noisy_labels, not_aug_inputs, epoch, true_labels=clean_labels)
→ model.observe(...)
→ loss.backward()
→ optimizer.step()
→ buffer.add_data(...)
→ model.meta_end_epoch()
→ model.meta_end_task()
→ dataset.evaluate(model, dataset)
→ dataset.log(args, logger, accs, ...)
→ 可选 save_mammoth_checkpoint()
→ 结束时 logger.write(vars(args))
```

`--inference_only 1` 不进入训练 batch loop，但仍会调用 `dataset.get_data_loaders()`、`model.meta_begin_task()`、`model.meta_end_task()`、`dataset.evaluate()`。

## 9. 核心数据流

CIFAR 样本流程：

```text
原始 CIFAR image/clean target
→ MyCIFAR10/MyCIFAR100.__getitem__()
→ augmented image, target, not_aug_img
→ MammothDatasetWrapper
→ store_masked_loaders()
→ 可选添加 true_labels=clean targets
→ build_noisy_labels() 生成或读取 noisy targets
→ train_dataset.targets = noisy_targets
→ task class mask
→ DataLoader batch
→ train_single_epoch(): inputs, labels, not_aug_inputs, extra_fields
→ model.meta_observe(..., true_labels=...)
→ backbone features
→ classifier logits
→ CE/OGC/replay/SAP 相关 loss
→ backward + optimizer.step()
→ buffer 保存 not_aug_inputs/noisy labels/可选 true_labels
```

已确认维度示例：

| 数据集 | 输入 | backbone features | logits |
|---|---|---|---|
| CIFAR10 + `resnet18` | `[B, 3, 32, 32]` | `[B, 512]` | `[B, 10]` |
| CIFAR100 + `resnet18` | `[B, 3, 32, 32]` | `[B, 512]` | `[B, 100]` |
| NTU60 + `efficient-gcn` | `[B, 3, 120, 25, 2]` | `[B, 256]`（默认 width=64） | `[B, 60]` |

## 10. 环境与依赖配置

依赖文件：

| 文件 | 状态 |
|---|---|
| `requirements.txt` | 主依赖，包含 `torch>=2.1.0`、`torchvision`、`kornia>=0.7.0`、`timm==0.9.8`、`pyyaml` 等 |
| `requirements-optional.txt` | 可选依赖，包含 `wandb`、`scikit-learn`、`pandas`、`transformers` 等 |
| `pyproject.toml` | Mammoth 包声明，`requires-python = ">=3.10"`，依赖版本更具体 |

当前 shell 低成本检查结果：

```bash
python --version
# 失败：当前 shell 中没有 python 命令

python3 --version
# Python 3.9.6

python3 -c "import torch"
# 失败：No module named 'torch'
```

推荐环境：

```bash
# 示例，不会自动安装；按服务器/本机 CUDA 或 MPS 情况调整
conda create -n nrgp-mammoth python=3.10
conda activate nrgp-mammoth
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

运行前检查：

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torchvision; print(torchvision.__version__)"
python -c "import sklearn, scipy, yaml, kornia; print('ok')"
```

设备支持由 `utils.conf.get_device()` 决定：优先 CUDA，未指定 GPU 时选择显存占用较低 GPU；无 CUDA 时尝试 Apple MPS；再退回 CPU。CPU 能否完成正式训练尚未验证。`--distributed dp` 支持 DataParallel；`ddp` 在代码中直接 `NotImplementedError`。

## 11. 数据集准备与目录结构

默认数据根目录由 `--base_path` 控制，默认 `./data/`。

### CIFAR-10

代码路径：`datasets/seq_cifar10.py`。

状态：`data/CIFAR10/cifar-10-batches-py` 已存在。

特点：

| 项目 | 值 |
|---|---|
| 注册名 | `seq-cifar10` |
| setting | `class-il` |
| 类数 | 10 |
| 任务数 | 5 |
| 每任务类别数 | 2 |
| 默认 backbone | `resnet18` |
| 默认 epoch/batch | 50 / 32 |
| 自动下载 | 是，`torchvision.datasets.CIFAR10(..., download=True)` |

### CIFAR-100

代码路径：`datasets/seq_cifar100.py`。

状态：当前未发现 `data/CIFAR100`。代码会尝试自动下载。

特点：

| 项目 | 值 |
|---|---|
| 注册名 | `seq-cifar100` |
| setting | `class-il` |
| 类数 | 100 |
| 默认任务数 | 10 |
| 默认每任务类别数 | 10 |
| 默认 backbone | `resnet18` |
| 默认 epoch/batch | 50 / 32 |
| 默认 scheduler | `multisteplr`，milestones `[35, 45]` |

### NTU60

代码路径：`datasets/seq_ntu60.py`，转换脚本 `scripts/prepare_ntu60_npz.py`。

状态：当前未发现 `data/NTU60_CS.npz`。

要求 `.npz` 包含 `x_train`、`y_train`、`x_test`、`y_test`。转换命令格式：

```bash
python scripts/prepare_ntu60_npz.py --source <skeleton_zip_or_dir> --output data/NTU60_CS.npz --split cs
```

### Food-101N

代码路径：`datasets/seq_food101n.py`，配置文件 `datasets/configs/seq-food101n/default.yaml`，研究与验证文档位于 `docs/food101n/`。

状态：当前未发现真实 Food-101N 或 Food-101 clean test 数据。本数据集不自动下载；应手动准备 Food-101N noisy train 和 Food-101 clean test。

特点：

| 项目 | 值 |
|---|---|
| 注册名 | `seq-food101n` |
| setting | `class-il` |
| 类数 | 101 |
| 任务数 | 5 |
| 每任务类别数 | `[20, 20, 20, 20, 21]` |
| 默认 backbone | `resnet18` |
| 默认 epoch/batch | 50 / 32 |
| 自动下载 | 否 |
| train label | Food-101N noisy class label |
| test label | Food-101 clean class label |

Food-101N 的训练标签已经是真实噪声标签，不应再设置 `--noise_rate` 触发 CIFAR 式合成噪声。若后续方法强制要求 `true_labels`，需要单独确认真实噪声协议下的处理方式，不得把 verification label 当成 clean class label。

不要随意删除 `data/`、`data/noisy_labels/`、`data/results*/` 或原始压缩包。删除或重新生成 noisy label cache 会影响复现实验可比性。

## 12. 噪声标签设置

参数定义在 `utils/args.py`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--noise_type` | `symmetric` | 支持别名 `sym`、`symm`、`asym`、`asymm` |
| `--noise_rate` | `0` | 范围 `[0, 1]` |
| `--disable_noisy_labels_cache` | `0` | 是否禁用 noisy label cache |
| `--cache_path_noisy_labels` | `noisy_labels` | 相对 `base_path()` 的缓存目录 |

真实实现：

```text
store_masked_loaders()
→ train_dataset.add_extra_return_field('true_labels', clean targets)
→ build_noisy_labels(train_dataset.targets, args)
→ train_dataset.targets = noisy_targets
```

缓存路径：

```text
data/noisy_labels/<dataset>/<noise_type>/<noise_rate_percent>/<seed>/noisy_targets
```

对称噪声：`get_symmetric_noise()` 对每类按 `round(class_count * noise_rate)` 随机抽样，替换为不同类，随机源为 `np.random.RandomState(args.seed)`。

非对称噪声：`get_asymmetric_noise()` 只对 CIFAR10/CIFAR100 显式支持。CIFAR10 映射在 `noisify_cifar10_asymmetric()`；CIFAR100 在每个 superclass 内翻到下一类。CIFAR100 `multiclass_noisify()` 内部固定 `RandomState(0)`，这是代码事实，可能与用户期望的 seed 行为不同，修改前必须评估可比性。

测试集不注入噪声。训练时 `labels` 是 noisy label；如果噪声开启，`true_labels` 作为 extra field 传给模型，用于日志、样本选择或 buffer 保存。

正式实验必须固定并记录 `--seed`，不要在不记录 seed 的情况下删除或重建 noisy label cache。

## 13. 类增量学习任务设置

类增量逻辑在 `ContinualDataset` 和 `store_masked_loaders()` 中。

| 数据集 | 任务设置 |
|---|---|
| `seq-cifar10` | 5 tasks，每任务 2 类 |
| `seq-cifar100` | 默认 10 tasks，每任务 10 类；另有 `20tasks.yaml`、`5tasks.yaml` |
| `seq-ntu60` | 6 tasks，每任务 10 类 |
| `seq-food101n` | 5 tasks，类别数 `[20, 20, 20, 20, 21]`，边界 `[0, 20, 40, 60, 80, 101]` |

`get_offsets(task_idx)` 根据 `N_CLASSES_PER_TASK` 和 task index 计算 `[start_c, end_c)`。`store_masked_loaders()` 按当前 task mask 训练集和测试集，并将每次测试 loader append 到 `dataset.test_loaders`。

class order：

| 参数 | 行为 |
|---|---|
| `--permute_classes 1` | 使用 seed 生成 `args.class_order` |
| `--custom_class_order` | 指定完整类别顺序 |
| `--custom_task_order` | 指定任务顺序，会设置 `permute_classes=True` |

当前推荐实验命令未显式设置 class order，默认按数据集原始类顺序。若引入 permutation 或自定义顺序，必须记录并同步更新本文件。

## 14. 模型、方法与关键模块

| 模块 | 路径 | 关键类/函数 | 作用 | 修改影响 |
|---|---|---|---|---|
| backbone 注册 | `backbone/__init__.py` | `register_backbone`, `get_backbone` | 根据 `--backbone` 构造网络 | 影响 logits 维度和 checkpoint |
| CIFAR ResNet | `backbone/ResNetBlock.py` | `resnet18`, `ResNet.forward` | CIFAR 主 backbone | 改 feature_dim/classifier 会影响 checkpoint |
| NTU EfficientGCN | `backbone/EfficientGCN.py` | `efficient_gcn` | NTU skeleton backbone | SAP 兼容性待验证 |
| 模型注册 | `models/__init__.py` | `get_model_names`, `get_model` | 扫描 `models/*.py` | 文件名/NAME 改动会影响 CLI |
| baseline | `models/er_ace_aer_abs.py` | `ErAceAerAbs.observe` | AER/ABS + buffer | 影响 baseline 和消融 |
| SAP only | `models/aer_sap.py` | `AerSap.apply_sap` | buffer clean/ambiguous split + SAP | 影响消融和 checkpoint |
| OGC+SAP | `models/ogc_sap.py` | `ErAceAerAbsOGC.observe` | OGC loss、queue、sample weighting、SAP | 当前主方法，高风险 |
| SAP 核心 | `utils/sap_core.py` | `activation_projection_based_unlearning` | SVD projection unlearning | 影响训练后 task boundary |
| SAP 挂载 | `utils/sap_model_utils.py` | `attach_sap_methods` | 给 ResNet 添加 activation/project 方法 | 非 ResNet 支持待确认 |
| buffer | `utils/buffer.py` | `Buffer`, `ABSSampling` | replay memory 和样本替换 | 影响公平性和 checkpoint |
| checkpoint | `utils/checkpoints.py` | `save_mammoth_checkpoint`, `mammoth_load_checkpoint` | 保存/加载模型、args、buffer | 高风险 |
| logger | `utils/loggers.py` | `Logger.write`, `log_accs` | 写入 `logs.pyd` | 影响结果汇总 |

`ogc_sap.py` 中存在 `disable_dynamic_threshold` / `fixed_prob_threshold` 分支，但当前 parser 未定义这两个参数；除非另有外部注入，不能把它们写成正式可用 CLI 参数。

## 15. 配置文件与参数优先级

真实优先级：

```text
--loadcheck 中的 args 作为默认补全
→ initial CLI 解析 dataset/model/backbone
→ model.get_parser() 定义模型参数
→ models/config/<model>.yaml 的 default/best（若存在）
→ datasets/configs/<dataset>/<dataset_config>.yaml 或 default.yaml
→ dataset 类的 @set_default_from_args 默认值
→ backbone/dataset 动态参数
→ management + experiment 参数
→ 最终 parser.parse_args(cmd)，命令行显式参数覆盖前面默认值
```

关键参数来源：

| 参数 | 来源 | 说明 |
|---|---|---|
| `dataset`, `model`, `backbone` | `utils.args.add_initial_args()`，配置也可提供 backbone | `backbone` 若命令行缺失可由 dataset/model config 提供 |
| `lr`, `batch_size`, `n_epochs` | dataset 默认、YAML、CLI | `lr` 最终必须有值 |
| `buffer_size`, `minibatch_size` | rehearsal parser + CLI | rehearsal 模型要求 `buffer_size` |
| `noise_type`, `noise_rate` | `utils.args.add_experiment_args()` | 命令行覆盖默认 |
| `base_path`, `results_path`, `checkpoint_path` | `utils.args.add_management_args()` | results_path 相对 base_path |
| OGC/SAP 参数 | `models/ogc_sap.py` / `models/aer_sap.py` parser | `aer_sap` 和 `ogc_sap` 当前未发现专属 YAML |

`models/config/er_ace_aer_abs.yaml` 存在 `seq-cifar100` 和 `seq-ntu60` 配置；`models/config/aer_sap.yaml` 和 `models/config/ogc_sap.yaml` 当前未发现。

## 16. 训练流程

进入项目目录：

```bash
cd mammoth_code
```

环境检查：

```bash
python --version
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

建议先做 smoke test，当前环境未验证：

```bash
python main.py --dataset seq-cifar10 --model ogc-sap --backbone resnet18 \
  --noise_rate 0.2 --noise_type symm \
  --sap_scale_coff 1000 --ogc_loss_weight 0.3 --ogc_low_conf_weight 0.5 \
  --ogc_buffer_penalty_coeff 1.5 --sap_retain_samples 400 \
  --n_epochs 1 --batch_size 32 --lr 0.03 --buffer_size 500 \
  --num_workers 0 --debug_mode 1 --disable_log 1 --stop_after 1
```

CIFAR10 正式命令以 `readme_latest.md` 为准，例如 symmetric 40%：

```bash
COMMON_ARGS="--backbone resnet18 --n_epochs 50 --batch_size 32 --lr 0.03 --buffer_size 500 --num_workers 4"
python main.py --dataset seq-cifar10 --model ogc-sap --noise_rate 0.4 --noise_type symm \
  --sap_scale_coff 2000 \
  --ogc_loss_weight 0.4 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 400 \
  --savecheck last --seed 0 ${COMMON_ARGS}
```

CIFAR100 推荐命令以 `readme_latest.md` 为准，注意当前未发现 `data/CIFAR100`，第一次运行可能触发下载。

不要在未确认设备、数据、seed、noise cache 和输出目录时启动完整训练。

## 17. 评估与测试流程

独立评估没有单独脚本；使用 `main.py` 加 `--loadcheck` 和 `--inference_only 1`。

`readme_latest.md` 中 CIFAR100 symmetric 60% 示例：

```bash
python main.py --loadcheck checkpoints/cifar100_symm_60_eval_best.pt \
  --inference_only 1 --start_from 9 --stop_after 10
```

代码事实：

| 项目 | 行为 |
|---|---|
| 训练中评估 | 每个 task 结束后自动 `dataset.evaluate()` |
| `--eval_epochs` | 可在 epoch 间评估 |
| 独立评估 | `--inference_only 1` 跳过训练，仍构建数据 loader 和 task 状态 |
| 指标 | Class-IL accuracy 与 Task-IL/masked accuracy；可选 FWT/BWT/forgetting |
| 保存 | `Logger.write()` 写入 `base_path/results_path/<setting>/<dataset>/<model>/logs.pyd` |

当前环境未执行 `main.py --help` 或评估命令，因为缺少 `torch`。

## 18. checkpoint 与恢复训练

保存位置由 `--checkpoint_path` 控制，默认 `./checkpoints/`。`--savecheck last` 只在最后一个 task 保存；`--savecheck task` 每个 task 保存。

命名规则在 `main.py`：

```text
<ckpt_name_prefix><model>_<dataset>_<dataset_config>_<buffer_size>_<n_epochs>_<timestamp>_<uuid>_<last|task>.pt
```

safe checkpoint 包含：

```text
model, optimizer, scheduler, args, results, buffer（若模型有 buffer_size）
```

加载流程：

```text
parse_args(): 如有 --loadcheck，先读取 checkpoint args 作为默认值
train(): 如有 --loadcheck，再加载 model/buffer/results
```

恢复训练/评估注意：

| 场景 | 说明 |
|---|---|
| 仅加载参数 | checkpoint 无 args 时会触发 `OnlyArgsError`，继续解析 CLI |
| 加载完整 checkpoint | 会加载 model state 和 buffer |
| `start_from` | 用于跳过前面 task 并建立 dataset/model task 状态 |
| `stop_after` | task 上限，end exclusive |
| 修改模型结构 | classifier、backbone、SAP/OGC state 改动可能导致不兼容 |
| 修改 buffer_size | `load_buffer()` 会检查 buffer size，可能失败 |

不要覆盖正式 checkpoint。`checkpoints/paused/` 是中断保存目录，清理前必须确认用途。

## 19. 实验结果与产物管理

| 产物 | 默认/现有位置 | 规则 |
|---|---|---|
| 数据 | `data/` | 不提交、不删除、不随意重建 |
| noisy labels | `data/noisy_labels/` | 正式实验必须固定 seed 和缓存 |
| 结果 | `data/results*/` | 调试与正式结果分目录 |
| checkpoint | `checkpoints/` | 不覆盖、不批量删除 |
| 日志 | `run_logs/` | 保留完整命令、PID、异常 |
| 辅助图/数组 | `output/` | 判断是否正式产物后再清理 |
| 外层报告 | `../*.md` | 只作为辅助，不在本任务中修改 |

建议实验命名：

```text
<dataset>_<noise-type><noise-rate>_<method>_seed<seed>_<date_or_runid>
```

如果使用项目已有 run id，应优先沿用现有命名，例如 `results_cifar10_symm40_readme_latest_local_retry2_20260703_231106`。

## 20. 复现实验规范

正式复现实验应固定：

| 项目 | 要求 |
|---|---|
| dataset / split | 不随意改变 |
| class order | 默认或自定义必须记录 |
| noisy labels | 固定 seed 和 cache |
| backbone | baseline 与改进方法一致 |
| epoch/batch/lr | 与论文或 `readme_latest.md` 一致 |
| buffer_size/minibatch_size | 同协议比较 |
| optimizer/scheduler | 同协议比较 |
| seed | 至少记录，正式结论应多 seed |
| metrics | Class-IL 为主，Task-IL 只能在论文未报告时作辅助 |

若因显存、MPS/CPU、num_workers 或 batch size 调整导致协议不同，结果必须标为资源受限复现或参数调整实验，不得称为严格复现。

随机性来源包括 Python、NumPy、PyTorch、CUDA、DataLoader worker、class permutation、noisy label generation、GMM/SAP clean/ambiguous split和 buffer sampling。

## 21. 主实验与消融实验规范

主实验比较完整方法与 baseline；消融实验只改变一个模块或一个明确因素。

`readme_latest.md` 给出的消融维度：

| 实验 | 命令核心 |
|---|---|
| Baseline AER/ABS | `--model er-ace-aer-abs` |
| SAP only | `--model aer-sap` |
| OGC only | `--model ogc-sap --enable_sap 0` |
| OGC + SAP | `--model ogc-sap --enable_sap 1` |

公平性要求：

| 项目 | 要求 |
|---|---|
| 数据和 noisy labels | 相同 |
| class order | 相同 |
| seed 集合 | 相同 |
| backbone/buffer/epoch/batch/lr | 除非实验变量，否则相同 |
| 评估代码 | 相同 |
| 结果 | 报告均值、标准差、完整配置 |

不得同时改变多个因素却称为单因素消融。

## 22. 代码修改规范

修改代码前必须：

1. 阅读本文件。
2. 阅读 `readme_latest.md`。
3. 定位真实入口和调用链。
4. 搜索动态注册关系、parser、配置和脚本引用。
5. 制定最小修改方案。

高风险区域：

| 区域 | 风险 |
|---|---|
| `datasets/utils/continual_dataset.py` | 影响 task 划分、noisy labels、验证集 |
| `datasets/utils/label_noise.py` | 影响噪声复现可比性 |
| `models/ogc_sap.py` | 影响主方法、loss、buffer、SAP |
| `utils/buffer.py` | 影响 replay 公平性和 checkpoint |
| `utils/checkpoints.py` | 影响旧权重兼容 |
| `utils/evaluate.py` / `utils/loggers.py` | 影响指标和论文对比 |
| `utils/args.py` / YAML | 影响默认参数和命令兼容 |

禁止用硬编码个人绝对路径修复问题。未经用户授权，不修改 `readme_latest.md`、普通 `README.md`、数据、checkpoint、日志、外层报告、压缩包或 Git 配置。

## 23. 文件与目录操作限制

禁止事项：

| 禁止操作 | 原因 |
|---|---|
| 删除 `data/` 原始数据 | 可能破坏复现 |
| 删除 `data/noisy_labels/` cache | 会改变 noisy label |
| 删除/覆盖 `checkpoints/` | 可能丢失正式权重 |
| 清空 `run_logs/` 或 `data/results*/` | 会丢失证据 |
| 解压外层备份覆盖当前代码 | 会破坏当前分支状态 |
| 批量格式化全仓库 | 工作树已有大量用户/历史改动 |
| 把 `.pyc` 当成源码 | 缓存不是维护对象 |
| 把 `other/` 当成当前注册模型 | 未在 `models/` 动态扫描 |

删除或清理前，必须列出候选文件、引用关系、理由和风险，等用户确认。

## 24. 常见问题与排查顺序

排查顺序：

```text
启动命令
→ 参数解析：utils/args.py, main.py
→ 模型/数据集注册：models/__init__.py, datasets/__init__.py
→ YAML 与 CLI 覆盖
→ 数据路径：base_path(), data/
→ noisy label cache
→ class order / task offsets
→ DataLoader num_workers
→ 输入和 logits 维度
→ checkpoint 参数兼容
→ loss 和 optimizer
→ SAP/OGC 阈值与 GMM
→ logger 和 results_path
```

常见问题：

| 问题 | 检查点 |
|---|---|
| `python` 不存在 | 激活 conda，或使用正确解释器 |
| `torch` 缺失 | 安装 `requirements.txt`，确认 Python >= 3.10 |
| CUDA 不可用 | `python -c "import torch; print(torch.cuda.is_available())"` |
| MPS/CPU 慢或结果不同 | 记录设备，不称严格复现 |
| 数据路径错误 | `--base_path`、`data/CIFAR10`、`data/CIFAR100`、`data/NTU60_CS.npz` |
| noisy labels 不一致 | seed、cache path、`disable_noisy_labels_cache` |
| `true_labels` 缺失 | 是否设置 `noise_rate > 0`；`observe()` 是否要求 true_labels |
| checkpoint 不兼容 | backbone、num_classes、buffer_size、model name |
| NTU60 命令失败 | `readme_latest.md` 中 `aer_ogc_sap` 未在 `models/` 注册 |
| 输出被覆盖 | 显式设置 `--results_path`、`--checkpoint_path`、`--ckpt_name` |
| loss NaN | OGC/SAP/lr/noise_rate/batch 数据 |

## 25. AI Agent 标准工作流程

每次任务按以下顺序：

1. 阅读根目录 `AGENTS.md`。
2. 如涉及子目录，检查局部 `AGENTS.md`。
3. 阅读 `readme_latest.md`。
4. 辅助参考 `README.md`，但不优先。
5. 阅读相关代码、配置、日志或脚本。
6. 搜索 `NAME`、`register_*`、`get_parser`、`add_argument`、`get_data_loaders`、`observe`。
7. 判断是否会影响实验公平性和旧 checkpoint。
8. 只修改必要文件。
9. 运行低成本验证。
10. 汇报修改文件、原因、验证、未验证项、是否需要更新 `AGENTS.md` 和是否建议同步 `readme_latest.md`。

## 26. AGENTS.md 更新规则

本文件不是一次性静态文件。以下变化必须检查并更新：

| 变化 | 是否必须更新 |
|---|---|
| 训练入口或评估入口变化 | 是 |
| `readme_latest.md` 位置或作用变化 | 是 |
| 新增/删除主要模型、baseline、改进方法 | 是 |
| 新增/删除数据集或数据格式变化 | 是 |
| 噪声生成、非对称映射、cache、seed 规则变化 | 是 |
| task 划分、class order、初始类别数、增量步长变化 | 是 |
| 默认 batch/lr/epoch/buffer/optimizer/scheduler 变化 | 是 |
| checkpoint 格式或恢复方式变化 | 是 |
| 输出目录、日志系统、指标计算变化 | 是 |
| 依赖、Python/PyTorch/CUDA 支持变化 | 是 |
| 待确认事项得到确认 | 是 |
| 发现本文档与代码不一致 | 是 |

通常不需要更新：一次性调试输出、单次 loss 数值、已撤销临时参数、纯格式改动、未形成长期规则的临时报错。

更新方式：

1. 只改受影响章节。
2. 保留仍有效内容。
3. 删除失效命令。
4. 新命令注明是否验证。
5. 更新待确认事项。
6. 增加变更记录。
7. 如果也影响 `readme_latest.md`，在汇报中建议同步，但未经用户授权不直接修改。

## 27. AGENTS.md 更新检查表

- [ ] 项目代码根目录仍然准确
- [ ] `readme_latest.md` 路径仍然准确
- [ ] 主要参考文档优先级仍然准确
- [ ] 项目目标仍然准确
- [ ] 当前项目状态已经更新
- [ ] 仓库目录结构与代码一致
- [ ] 核心入口与代码一致
- [ ] 训练调用链仍然准确
- [ ] 评估调用链仍然准确
- [ ] 环境版本仍然有效
- [ ] 安装命令仍然有效
- [ ] 数据目录说明仍然有效
- [ ] 噪声标签说明仍然准确
- [ ] 类增量任务划分仍然准确
- [ ] 配置参数及优先级仍然准确
- [ ] 训练命令仍然有效
- [ ] 评估命令仍然有效
- [ ] checkpoint 说明仍然准确
- [ ] 输出目录说明仍然准确
- [ ] 复现实验规范未被破坏
- [ ] 已知问题已经更新
- [ ] 待确认事项已经更新
- [ ] 没有写入密码、Token 或个人绝对路径
- [ ] 已添加变更记录
- [ ] 已判断是否需要同步更新 `readme_latest.md`

## 28. 待确认事项

| 编号 | 待确认内容 | 当前依据 | 建议验证方式 | 状态 |
|---|---|---|---|---|
| T1 | 当前推荐 Python/Conda 环境 | 当前 shell `python` 不存在，`python3` 缺 torch；历史日志显示曾用独立环境 | 激活项目环境后执行环境检查命令 | 待确认 |
| T2 | `main.py --help` 输出是否完整 | 代码 parser 已读，但当前环境无法导入 torch | 安装依赖后执行 `python main.py --help` | 当前环境未完成验证 |
| T3 | CIFAR100 数据准备状态 | 未发现 `data/CIFAR100`，但有 CIFAR100 checkpoint | 检查服务器/外层数据包或运行只读数据路径检查 | 待确认 |
| T4 | NTU60 是否可跑通 | `seq-ntu60` 存在，数据缺失；`readme_latest.md` 使用未注册 `aer_ogc_sap` | 准备 `data/NTU60_CS.npz` 后测试当前注册模型 | 代码与文档存在差异 |
| T5 | `other/aer_ogc_sap.py` 是否为历史模型 | 文件在 `other/`，不被 `models/__init__.py` 扫描 | 查 Git 历史或用户说明 | 疑似历史遗留，尚未完全验证 |
| T6 | `models/ogc_sap_old.py` 是否仍被任何脚本使用 | 文件存在但主命令使用 `models/ogc_sap.py` | 全仓搜索脚本和日志引用 | 疑似历史遗留，尚未完全验证 |
| T7 | checkpoint 恢复训练是否能自动从正确 task 继续 | 代码需要 `start_from` 建 task 状态，未实测 | 用小型 debug checkpoint 验证恢复 | 尚未验证 |
| T8 | CIFAR10 asymmetric noisy label cache | 当前只发现 symmetric 20/40 cache | 检查运行日志或重新生成前先备份/记录 seed | 待确认 |
| T9 | 多 seed 正式结果 | `readme_latest.md` 建议 seed 0/152，结果记录不完整 | 汇总 `logs.pyd` 和外层报告 | 待确认 |
| T10 | `readme_latest.md` 是否需要同步修正 NTU60 命令 | 本次任务不允许修改 | 向用户建议单独授权更新 | 待确认 |
| T11 | Food-101N 真实数据路径和 Kaggle 解压结构 | 当前仅完成代码接入，未发现本机数据 | 准备 Food-101N noisy train 与 Food-101 clean test 后运行数据完整性验证 | 待确认 |
| T12 | Food-101N 与强制 `true_labels` 模型的协议兼容 | `er-ace-aer-abs.observe()` 要求 `true_labels`，但 Food-101N train 无完整 clean class label | 用户确认后决定修改模型签名、限制模型或设计显式兼容字段 | 待确认 |

## 29. 变更记录

| 日期 | 修改内容 | 修改原因 | 验证情况 |
|---|---|---|---|
| 2026-07-14 | 新增 `seq-food101n` 数据集接入说明、非均匀任务划分和待确认项 | Food-101N 数据集支持需要长期记录数据格式、真实噪声协议和 101 类 task split | 已完成 `py_compile`；当前 shell 缺 `torch`、`torchvision`、`yaml`，未完成 import/真实数据验证 |
| 2026-07-14 | 新建项目级 `AGENTS.md` | 建立 `mammoth_code` 的 AI Agent 长期操作规范，并明确 `readme_latest.md` 为主要参考文档 | 部分验证：已只读检查代码、配置、数据、日志、checkpoint；当前 shell 缺依赖，未执行训练/评估入口 |
