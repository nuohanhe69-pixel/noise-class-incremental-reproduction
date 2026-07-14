# Food101N Phase 6 Training Integration

## 本阶段目标

在已经完成 `seq-food101n` 数据集接入后，本阶段进入训练前集成：

1. 给出 Food101N 本地数据完整性检查脚本。
2. 将 Food101N 的 debug smoke 与正式训练模板同步到 `readme_latest.md`。
3. 处理真实 noisy train label 与 `true_labels` 接口的兼容问题。

## 已完成改动

| 文件 | 改动 |
|---|---|
| `scripts/validate_food101n_dataset.py` | 新增 Food101N/Food-101 本地数据检查脚本 |
| `models/er_ace_aer_abs.py` | `observe(..., true_labels=None)`，允许真实噪声数据集不提供 clean train label |
| `readme_latest.md` | 新增 Food101N 数据检查、debug smoke、正式训练模板和消融模板 |
| `docs/food101n/05_food101n_training_integration.md` | 记录本阶段决策 |

## `true_labels` 兼容策略

Food101N train label 是真实 noisy class label。官方提供的 verification label 只表示“当前 noisy label 是否被人工验证为正确”，不是完整 clean class label。

因此本阶段不伪造 `true_labels`，也不把 verification label 当成 clean class label。实际策略是：

1. `er_ace_aer_abs.observe()` 将 `true_labels` 改为可选参数。
2. `aer_sap` 继承该实现，因此同步兼容。
3. `ogc_sap.observe()` 原本已经支持 `true_labels=None`，并且写入 buffer 时也只在存在 `true_labels` 时保存。
4. 后续如果某个评估或日志需要 noisy/clean 对照，只能在拥有可靠 clean train label 的合成噪声数据集上启用。

## 数据检查脚本用法

默认推荐：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N_release \
  --test-list ../food-101/meta/test.txt \
  --test-images-dir ../food-101/images \
  --classes-file ../food-101/meta/classes.txt \
  --strict
```

如果项目环境中已经安装 `torch`、`torchvision`，可以增加：

```bash
--check-import
```

如果还要真实读取一个 train/test batch，可以继续增加：

```bash
--check-batch
```

脚本会检查：

| 检查项 | 说明 |
|---|---|
| metadata 可读 | 支持 JSON/TXT/CSV/TSV 与 class-folder 扫描 |
| 类别数 | 默认必须为 101 |
| label 范围 | 默认必须为 `0..100` |
| 图片路径 | 默认统计缺失路径；`--strict` 下遇到缺失直接失败 |
| task split | 固定检查 `[20, 20, 20, 20, 21]` 总和为 101 |
| Mammoth import | 仅在 `--check-import` 时构造 `SequentialFood101N` loader |
| batch 读取 | 仅在 `--check-import --check-batch` 时读取一个 train/test batch |

## 当前阻塞

本机仍未发现真实 Food101N/Food-101 数据，也没有当前 shell 可用的 `torch`、`torchvision` 环境。因此本阶段没有启动训练，也没有产出 Food101N accuracy。

准备好数据和环境后，下一步顺序应为：

1. 跑 `scripts/validate_food101n_dataset.py --strict`。
2. 跑 `scripts/validate_food101n_dataset.py --strict --check-import`。
3. 视需要跑 `scripts/validate_food101n_dataset.py --strict --check-import --check-batch`。
4. 跑 `readme_latest.md` 中的 Food101N debug smoke。
5. 再跑正式 OGC+SAP 与消融实验。
