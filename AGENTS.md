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
| Baseline + DGC | `dgc` | `models/dgc.py` | 在 AER/ABS 上加入动态梯度裁剪；当前主方法 |
| Baseline + DGC + SAP | `dgc-sap` | `models/dgc_sap.py` | 每 Task 训练后从当前 Task 筛选持久 trusted reference，对 ResNet18 `layer3/layer4` 执行事务式 pre-projection |
| ER / ER-ACE | `er`, `er-ace` | `models/er.py`, `models/er_ace.py` | Mammoth 原有/基础 replay baseline，可辅助比较 |

旧 CBP/SAP 实现已于 2026-08-12 从活跃代码和历史副本中删除，删除前版本可从 Git 历史恢复。2026-08-14 已在独立 `dgc-sap` 入口重新接入论文式 SAP；当前等待 RTX 4090 上的 CIFAR100 3-Task smoke，通过后才运行完整实验。

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
| 参数调整实验 | 为追指标调整 DGC 超参 | 已有多份 CIFAR10 调参汇总 |
| baseline 实验 | `er-ace-aer-abs` 等 baseline 单独运行 | 部分记录存在，需按日志逐项确认 |
| 完整方法实验 | `dgc` | CIFAR10 多个历史结果目录与日志存在；新入口需重新核验 |
| 消融实验 | `er-ace-aer-abs` 与 `dgc` 对比 | `readme_latest.md` 给出命令；完整执行状态待确认 |

当前已开展的证据主要集中在 CIFAR10；CIFAR100 有 checkpoint 和外层整理文档；NTU60 目前更多是代码和结果文档线索，数据文件不在当前仓库。

## 5. 当前项目状态

以下状态已于 2026-08-14 更新。

| 项目 | 状态 | 证据 |
|---|---|---|
| Git 仓库 | 已确认 | 当前目录是 Git 根目录，正在 `SAP` 分支实现和验证 `dgc-sap` |
| 工作树 | 本地 SAP 实现已完成 + Oracle 分支已完成 | SAP 数学、ResNet hook、Robust GMM/reference、任务边界事务与诊断已分阶段验证；2026-08-19 新增 oracle 上限分支（`--sap_oracle_reference`，仅投影 classifier）并完成本机 debug 冒烟；下一步为 RTX4090 CIFAR100 3-Task smoke，oracle 实验先跑 CIFAR10 symm20 scale 预实验 {10,30,100,1000} |
| 当前 shell 环境 | 已确认 | `nrgp-mammoth` 提供 Python 3.10、PyTorch 与测试依赖 |
| 依赖声明 | 已存在 | `requirements.txt`、`requirements-optional.txt`、`pyproject.toml` |
| CIFAR-10 数据 | 已准备 | `data/CIFAR10/cifar-10-batches-py` 存在 |
| CIFAR-100 数据 | 未发现 | 未发现 `data/CIFAR100` |
| NTU60 数据 | 未发现 | 未发现 `data/NTU60_CS.npz` |
| Food-101N 数据 | 未发现 | 未发现 `data/Food-101N`、`data/Food-101N_release`；当前正式协议需要 Food101N 图片目录 + NTD/PuriDivER `tasks/Food-101N` split |
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
│   ├── dgc.py                     # Baseline + DGC
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
│   ├── loss_trace.py            # iteration 级逐样本 loss 与四组曲线记录
│   └── ...
├── scripts/
│   ├── prepare_ntu60_npz.py
│   └── plot_loss_trace.py       # 绘制 noisy/hard-old/easy-old/new 四线图
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
| `models/dgc.py` | 当前主改进方法 | 是 | `models.get_model()` | 高 | 待本轮验证 | Baseline + DGC，parser 保留 `ogc_*` 参数以兼容已有实验配置 |
| `models/dgc_sap.py` | DGC + 新 SAP | 是 | `models.get_model()` | 高 | 真实 CIFAR10 验证已通过 | 完成后评分、按类 Robust GMM、持久 reference、事务式 `layer3/layer4` pre-projection、dry-run/回滚和诊断 |
| `utils/sap.py` / `utils/sap_reference.py` / `utils/sap_runtime.py` | SAP 核心、trusted reference 与任务边界运行时 | 是（仅 `dgc-sap`） | `models/dgc_sap.py` | 高 | 31 个单元测试+真实 CIFAR10 分阶段验证 | 只投影 ResNet18 `layer3/layer4` 的10个 Conv2d；每层最多采样 patch 数由 CLI 控制 |
| `utils/buffer.py` | replay buffer 和采样策略 | 是 | rehearsal models | 高 | 已阅读 | 支持 reservoir/lars/labrs/abs/balancoir |
| `utils/loss_trace.py` | 可选的逐样本 loss 和四组 iteration 曲线记录 | 启用参数时 | AER/ABS、DGC | 中 | 单元与 debug 训练已验证 | 默认关闭；要求 synthetic noise 的 `true_labels` |
| `scripts/plot_loss_trace.py` | 从 `iteration_curves.csv` 生成四线图 | 手工运行 | 用户/实验脚本 | 低 | 语法已验证 | 需要可选依赖 matplotlib |
| `utils/checkpoints.py` | checkpoint 保存/加载 | 是 | `main.py`, `training.py` | 高 | 真实 CIFAR10 非空 buffer 跨 Task 恢复已验证 | safe checkpoint 保存 args/results/buffer；buffer 同时保存累计见样本数与采样器状态；`dgc-sap` 额外保存/恢复 `sap_state` |
| `readme_latest.md` | 当前主要运行说明 | 是，文档参考 | 人/Agent | 中 | 已阅读 | 不得直接覆盖 |
| `README.md` | 上游 Mammoth + 历史片段 | 辅助 | 人/Agent | 中 | 已阅读 | 有旧命令与当前代码差异 |
| `checkpoints/1.py` | 硬编码权重转换脚本 | 否/辅助 | 手工运行 | 高 | 已阅读 | 不要直接执行或改源/目标路径 |

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
→ CE/DGC/replay 相关 loss
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

代码路径：`datasets/seq_food101n.py`，配置文件 `datasets/configs/seq-food101n/default.yaml`，研究与验证文档位于 `docs/food101n/`。用户已要求后续 Food-101N/论文相关总结统一以 `.md` 文件交付。

状态：当前未发现真实 Food-101N 图片数据；已只读扫描项目 `data/`、`Documents`、`Downloads`、`Desktop` 的常见路径，未发现 Food101N 图片目录。NTD/PuriDivER 仓库提供 `tasks/Food-101N` split JSON，可通过 `git clone --depth 1 https://github.com/wish44165/ntd.git external/ntd` 获取。用户已确认 Food101N 正式复现采用论文/AER/NTD 接近的 `52,867 train / 4,741 test` split，不直接扫描完整约 310k Food101N，也不使用完整 Food-101 官方 25,250 clean test 作为首轮正式复现。

特点：

| 项目 | 值 |
|---|---|
| 注册名 | `seq-food101n` |
| setting | `class-il` |
| 类数 | 101 |
| 任务数 | 5 |
| 每任务类别数 | `[20, 20, 20, 20, 21]` |
| 默认 backbone | `resnet34` |
| 默认 epoch/batch | 20 / 32 |
| 自动下载 | 否 |
| train label | Food-101N noisy class label |
| test label | NTD/PuriDivER Food-101N `4,741` test split label；不是完整 Food-101 25,250 clean test |

Food-101N 的训练标签已经是真实噪声标签，不应再设置 `--noise_rate` 触发 CIFAR 式合成噪声。当前 `er-ace-aer-abs` 与 `dgc` 均允许 `true_labels=None`。不得把 verification label 当成 clean class label。

本地数据检查入口：

```bash
python scripts/prepare_food101n_metadata.py --output-root data/Food-101N \
  --train-list <source_train_split> --test-list <source_test_split> \
  --train-images-dir <source_train_images> --test-images-dir <source_test_images> \
  --classes-file <source_classes_file> --strict --overwrite
python scripts/validate_food101n_dataset.py --root data/Food-101N_release --strict
python scripts/validate_food101n_dataset.py --root data/Food-101N_release --strict --check-import --check-batch
```

正式 split 验证优先使用显式 metadata：

```bash
python scripts/validate_food101n_dataset.py --root data/Food-101N \
  --train-list meta/train.tsv --test-list meta/test.tsv \
  --images-dir images --classes-file meta/classes.txt --strict
```

验证脚本默认要求 train/test 样本数为 `52,867` 和 `4,741`；如只是临时检查非正式 split，可显式传 `--expected-train-records 0 --expected-test-records 0` 关闭样本数检查。

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
| NTU EfficientGCN | `backbone/EfficientGCN.py` | `efficient_gcn` | NTU skeleton backbone | DGC 兼容性待验证 |
| 模型注册 | `models/__init__.py` | `get_model_names`, `get_model` | 扫描 `models/*.py` | 文件名/NAME 改动会影响 CLI |
| baseline | `models/er_ace_aer_abs.py` | `ErAceAerAbs.observe` | AER/ABS + buffer | 影响 baseline 和消融 |
| DGC | `models/dgc.py` | `DGC.observe` | DGC loss、queue、sample weighting | 当前主方法，高风险 |
| buffer | `utils/buffer.py` | `Buffer`, `ABSSampling` | replay memory 和样本替换 | 影响公平性和 checkpoint |
| checkpoint | `utils/checkpoints.py` | `save_mammoth_checkpoint`, `mammoth_load_checkpoint` | 保存/加载模型、args、buffer | 高风险 |
| logger | `utils/loggers.py` | `Logger.write`, `log_accs` | 写入 `logs.pyd` | 影响结果汇总 |

`dgc.py` 中存在 `disable_dynamic_threshold` / `fixed_prob_threshold` 分支，但当前 parser 未定义这两个参数；除非另有外部注入，不能把它们写成正式可用 CLI 参数。

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
| DGC 参数 | `models/dgc.py` parser | 为保持实验可比性，CLI 参数仍使用 `ogc_*` 前缀；当前无专属 YAML |

`models/config/er_ace_aer_abs.yaml` 存在 `seq-cifar100` 和 `seq-ntu60` 配置；`models/config/dgc.yaml` 当前未发现。

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
python main.py --dataset seq-cifar10 --model dgc --backbone resnet18 \
  --noise_rate 0.2 --noise_type symm \
  --ogc_loss_weight 0.3 --ogc_low_conf_weight 0.5 \
  --ogc_buffer_penalty_coeff 1.5 \
  --n_epochs 1 --batch_size 32 --lr 0.03 --buffer_size 500 \
  --num_workers 0 --debug_mode 1 --stop_after 1 \
  --results_path /tmp/dgc_smoke_results --checkpoint_path /tmp/dgc_smoke_checkpoints
```

当前 `--disable_log 1` 会在 `utils/training.py` 中触发未初始化 `logger` 的既有错误；smoke 暂时使用 `/tmp` 结果目录隔离测试产物。

CIFAR10 正式命令以 `readme_latest.md` 为准，例如 symmetric 40%：

```bash
COMMON_ARGS="--backbone resnet18 --n_epochs 50 --batch_size 32 --lr 0.03 --buffer_size 500 --num_workers 4"
python main.py --dataset seq-cifar10 --model dgc --noise_rate 0.4 --noise_type symm \
  --ogc_loss_weight 0.4 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
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

2026-08-14 起，新 safe checkpoint 的 `buffer` 还包含版本化恢复状态：

```text
num_seen_examples, sample_selection_strategy, sample_selection_state
```

其中 ABS/LARS 会保存 `importance_scores`，避免恢复后替换概率和采样权重被重置。旧 safe checkpoint 不含 `num_seen_examples`，只允许 `--inference_only 1` 评估；训练续跑会 fail-closed 并明确报错，不能再静默把累计计数设为 buffer 容量。

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
| 修改模型结构 | classifier、backbone、DGC state 改动可能导致不兼容 |
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
| loss trace | `output/loss_trace/<dataset>_<model>_<run-id>/` | 保存 metadata、逐样本 CSV、iteration 曲线和可选图片 |
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

随机性来源包括 Python、NumPy、PyTorch、CUDA、DataLoader worker、class permutation、noisy label generation、DGC 的 GMM 和 buffer sampling。

## 21. 主实验与消融实验规范

主实验比较完整方法与 baseline；消融实验只改变一个模块或一个明确因素。

`readme_latest.md` 给出的消融维度：

| 实验 | 命令核心 |
|---|---|
| Baseline AER/ABS | `--model er-ace-aer-abs` |
| Baseline + DGC | `--model dgc` |

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
| `models/dgc.py` | 影响主方法、loss、queue 和 buffer |
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
→ DGC 阈值与 GMM
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
| NTU60 命令失败 | 检查 `dgc` 与 `efficient_gcn` 的接口兼容性及数据文件 |
| 输出被覆盖 | 显式设置 `--results_path`、`--checkpoint_path`、`--ckpt_name` |
| loss NaN | DGC/lr/noise_rate/batch 数据 |

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
| T1 | 当前推荐 Python/Conda 环境 | `nrgp-mammoth` 已确认 Python 3.10.20、torch 2.12.1；未安装 matplotlib | 正式运行前激活该环境，绘图时安装 optional dependency | 训练环境已确认，绘图依赖待安装 |
| T2 | `main.py --help` 输出是否完整 | 已在 `nrgp-mammoth` 中确认 loss trace 及 `dgc-sap` 参数进入 parser | 后续新增参数时继续做 CLI smoke test | 已验证本次新增参数 |
| T3 | CIFAR100 数据准备状态 | 未发现 `data/CIFAR100`，但有 CIFAR100 checkpoint | 检查服务器/外层数据包或运行只读数据路径检查 | 待确认 |
| T4 | NTU60 是否可跑通 | `seq-ntu60` 存在，数据缺失；文档已改用注册模型 `dgc` | 准备 `data/NTU60_CS.npz` 后测试 DGC | 数据待准备 |
| T7 | checkpoint 恢复训练是否能自动从正确 task 继续 | `start_from` 建 task 状态；新 safe checkpoint 保存 buffer 累计计数和采样器状态 | 已用真实 CIFAR10 非空 ABS buffer 从 Task 1 checkpoint 恢复 Task 2，计数48→96且旧/新 Task 各48张 | 已验证；旧 safe checkpoint 仅允许推理 |
| T8 | CIFAR10 asymmetric noisy label cache | 当前只发现 symmetric 20/40 cache | 检查运行日志或重新生成前先备份/记录 seed | 待确认 |
| T9 | 多 seed 正式结果 | `readme_latest.md` 建议 seed 0/152，结果记录不完整 | 汇总 `logs.pyd` 和外层报告 | 待确认 |
| T10 | `readme_latest.md` 是否需要同步修正 NTU60 命令 | 本次任务不允许修改 | 向用户建议单独授权更新 | 待确认 |
| T11 | Food-101N 真实数据路径和 Kaggle 解压结构 | split 已确认采用 NTD/PuriDivER `52,867/4,741`；当前仍未发现本机真实图片数据 | 下载 Food101N 图片；clone NTD/PuriDivER task split；运行 `prepare_food101n_metadata.py` 和验证脚本 | split 获取方式已确认，图片数据路径待确认 |
| T12 | Food-101N 与强制 `true_labels` 模型的协议兼容 | Food-101N train 无完整 clean class label | 已将 `er-ace-aer-abs.observe()` 改为 `true_labels=None`；不伪造 clean train label | 已处理 |
| T13 | CIFAR100 DGC+SAP 的最终 patch 上限和完整结果 | 本地真实 CIFAR10 验证已通过；标准 ResNet18 最大 patch 维度4608、服务器为 RTX 4090 | 先在服务器运行 `bash scripts/run_cifar100_dgc_sap_server.sh smoke`；资源或机制失败才回退10000/5000，通过后单独运行 `full` | 待服务器验证 |

## 29. 变更记录

| 日期 | 修改内容 | 修改原因 | 验证情况 |
|---|---|---|---|
| 2026-08-14 | safe checkpoint 增加版本化 buffer 累计计数/采样器状态；旧格式训练恢复 fail-closed；SAP candidate gate 对 NaN/Inf、诊断样本不足和已见 replay Task 缺失 fail-closed | 修复中断恢复后 `num_seen_examples` 重置导致旧 buffer 被覆盖，以及稀疏旧 Task 产生 NaN 后安全门错误放行 | 48 个 unittest、`py_compile`、真实 Task 2 历史 buffer 只读兼容检查、真实 CIFAR10 空/非空 buffer 跨 Task debug 恢复通过 |
| 2026-08-14 | 新增独立 `dgc-sap`：论文式 SAP 数学、ResNet18 `layer3/layer4` pre-hook/Gram 投影、seen-class CE + Robust GMM、持久 reference、事务/dry-run/回滚、checkpoint、投影/replay 诊断和 CIFAR100 服务器脚本 | 在干净 Baseline+DGC 上重新接入可验证 SAP，且不恢复旧 CBP/旧 SAP | 31 个单元测试通过；真实 CIFAR10 数据验证 10 层 patch/投影、Symm40 checkpoint GMM（30 references、fallback=0、诊断纯度1.0）、dry-run/提交/回滚、500张 replay 三档诊断、5个 test Task 即时准确率 delta 和 safe checkpoint恢复；CIFAR100 smoke/full 待 RTX4090 运行 |
| 2026-08-12 | 删除旧 CBP/SAP 活跃与历史代码，将原主模型重构为注册名 `dgc` 的纯 Baseline+DGC | 用户审核通过完整删除范围，先建立可验证的干净 DGC 基线，再重新接入 SAP | `py_compile`、注册/CLI 检查、3 个 loss trace unittest、Seq-CIFAR10 两任务 debug smoke 通过；结果与删除前 `ogc-sap --enable_sap 0` 完全一致 |
| 2026-07-31 | 新增 iteration 级逐样本 CE loss、动态 noisy/hard-old/easy-old/new 聚合、CSV 输出和绘图脚本 | 用户要求将 Figure 1 扩展为四线图，并保留 task/epoch/iteration/sample/buffer/class/loss 级数据 | 3 个 unittest 通过；`er-ace-aer-abs` 与当时的旧主模型均完成 Seq-CIFAR10 两任务 debug 端到端验证；绘图脚本因当前环境缺 matplotlib 仅完成语法/CLI 验证 |
| 2026-07-14 | 拉取 NTD `tasks/Food-101N`，生成本机 `data/Food-101N/meta/train.tsv/test.tsv/classes.txt`，并完成离线 metadata 验证 | 用户要求先解决除数据下载外的全部问题；split metadata 可由 NTD/PuriDivER 提供，图片本体仍需用户下载 | 已验证 records=52,867/4,741、classes=101、label range=0..100；图片路径检查因未下载图片仍待执行 |
| 2026-07-14 | 新增 `scripts/prepare_food101n_metadata.py`，并记录本地 Food101N 数据只读扫描结果 | 下一步需要把 AER/NTD/Food101N 原始 split 转成项目标准 `train.tsv/test.tsv/classes.txt` | 已完成脚本语法检查；当前本机仍未发现真实 Food101N 数据 |
| 2026-07-14 | 确认 Food101N 采用论文/AER/NTD 接近的 `52,867 train / 4,741 test` split，并同步默认配置和验证脚本 | 用户明确选择该 split；需要避免误扫完整约 310k Food101N，且默认配置应对齐 ResNet34/20 epochs | 已更新配置与验证逻辑；真实数据仍待准备后验证 |
| 2026-07-14 | 新增原文 Food101N 协议与代码实现说明文档；后续 Food101N/论文相关总结统一输出为 `.md` 文件 | 用户要求所有相关总结以 Markdown 文件交付，并需要记录原文协议、参考论文源码和实现解释 | 已阅读用户提供 PDF，核对 Food101N 关键页和当前实现；真实数据仍待验证 |
| 2026-07-14 | 新增 Food-101N 数据检查脚本、训练命令模板，并放宽 `er-ace-aer-abs.observe()` 的 `true_labels` 参数 | 进入 Food-101N 训练接入阶段，真实 noisy train 数据不应要求 clean train label | 已完成 `py_compile` 级别验证；真实数据和 torch/torchvision 环境仍待验证 |
| 2026-07-14 | 新增 `seq-food101n` 数据集接入说明、非均匀任务划分和待确认项 | Food-101N 数据集支持需要长期记录数据格式、真实噪声协议和 101 类 task split | 已完成 `py_compile`；当前 shell 缺 `torch`、`torchvision`、`yaml`，未完成 import/真实数据验证 |
| 2026-07-14 | 新建项目级 `AGENTS.md` | 建立 `mammoth_code` 的 AI Agent 长期操作规范，并明确 `readme_latest.md` 为主要参考文档 | 部分验证：已只读检查代码、配置、数据、日志、checkpoint；当前 shell 缺依赖，未执行训练/评估入口 |

# 30. 范围约束

=== 范围约束(约束你提议什么修法,不约束你找什么)===
凡是这里真的有问题,都要报——包括听起来罕见但本项目确实会产生的情况。
然后把修法收在范围内:
1. 这不是一篇安全攻防论文。可以校验,禁止过度防御。除非本项目另有说明,默认操作者是
   自己机器上的合作者;如果它真有对手,它会写明,以那个范围为准。
2. 不要加哈希/校验和/指纹,除非它替代了一个实质上更贵的操作,并且结果会改变下一步做什么。
3. 禁止防御性脚手架:不为这里不会发生的情况加 feature flag、迁移框架、兼容层、包装层。
4. 禁止钻牛角尖:冷门编码、符号链接竞态、RTL 文本、毫秒级竞态一律不在范围内,
   除非该情况经由本项目**受支持的用法**可达——它的文档示例、它公开的接口、它真实的
   数据。可达即可,不需要你复现出来;但"理论上构造得出"不算。
5. 该判断的地方就判断,不要换成评分表、检查清单,或对已经定论的东西再跑一遍校验。
6. 以上都不覆盖用户、本项目自己的约定、或更高优先级规则明确要求的安全、迁移、校验与
   审阅。那些是被要求的,是活儿本身,不算范围外。
已经见过的形状,供你校准。是例子不是清单——一个真问题不会因为"长得像其中一条"就被驳回:
  H  为了比对两个表格的差异,给每一行都算哈希——直接比单元格就能回答
  H  写下一堆校验和文件,而没有任何代码会去读它们
  E  给一个没有用户、没有部署的应用做账号安全加固
  R  用一整夜对自己的补丁反复审计,而功能一行没写
  R  一个对任何提交都给不通过的审阅者
  O  一层守卫的理由是上一层守卫,而不是需求
另有两种长得像上面、但不是的。这些要报:
  ✓  用摘要比对来跳过重读一个你已经有的大文件
  ✓  本项目自己的文档示例就会产生的那种"听起来罕见"的输入
跑任何检查之前先回答:这次运行会检测出什么具体的失败?真出现了我下一步会做什么不同的事?
答不上来就别跑。
对的就说对。不要为了交差硬找问题。