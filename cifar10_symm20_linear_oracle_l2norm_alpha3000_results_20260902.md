# CIFAR10 Symm20 Power-Normalized Linear Oracle SAP（α=3000）实验结果

## 实验目的

本实验与此前未进行 activation L2 normalization、`sap_oracle_scale=3000` 的 Linear Oracle SAP 实验构成单变量对照。唯一算法变化是在构造 SAP activation matrix / Gram 前，对每个进入最终 Linear 的 512 维 feature 做逐样本 L2 normalization：

```text
Linear activation
→ sample-wise L2 normalization
→ ordinary Gram
→ original SAP
→ full Linear projection
```

L2 normalization 仅用于 SAP 计算，不修改网络正常 forward 到 Linear 的 feature。

## 固定配置

| 参数 | 值 |
|---|---:|
| Dataset | CIFAR10 / `seq-cifar10` |
| Noise | symmetric 20% |
| Backbone | ResNet18 |
| Tasks | 5（每 Task 2 类） |
| Epochs | 50 / Task |
| Batch size | 32 |
| Learning rate | 0.03 |
| Buffer size | 500 |
| Workers | 4 |
| `ogc_loss_weight` | 0.30 |
| `ogc_low_conf_weight` | 0.30 |
| `ogc_buffer_penalty_coeff` | 2.0 |
| `sap_oracle_reference` | 1 |
| `sap_oracle_scale` | 3000 |
| Seed | 0 |
| Device | Apple MPS |

未启用 class-balanced Gram、seen-row projection、GMM、safety gate、confidence/loss 筛选或 sample sampling。SAP reference 保持为当前 Task 全部 oracle clean samples，加 epoch 50 时 buffer 中全部 oracle clean samples；SAP 仍只作用于最终 Linear。

## 最终结果

- 最终 Class-IL：**63.68%**
- 最终 Task-IL：**90.47%**
- Class-IL raw accuracies：`[63.70, 50.60, 62.05, 73.55, 68.50]`
- Task-IL raw accuracies：`[87.05, 85.10, 90.40, 94.65, 95.15]`
- 训练进程正常退出：exit code 0
- 未出现 `Traceback`、`SAP_FAILED`、`RuntimeError` 或 `Exception`

## 各 Task SAP 前后准确率

| Boundary | Seen-task accuracy before → after | Seen average before → after |
|---|---|---:|
| Task 1 | T1 96.75% → 96.65% | 96.75% → 96.65% |
| Task 2 | T1 84.55% → 84.40%; T2 83.75% → 84.10% | 84.15% → 84.25% |
| Task 3 | T1 83.85% → 83.90%; T2 63.05% → 64.75%; T3 73.95% → 73.40% | 73.6167% → 74.0167% |
| Task 4 | T1 73.00% → 72.15%; T2 55.30% → 56.30%; T3 58.25% → 58.65%; T4 76.35% → 76.20% | 65.725% → 65.825% |
| Task 5 | T1 64.60% → 63.70%; T2 51.35% → 50.60%; T3 61.55% → 62.05%; T4 72.70% → 73.55%; T5 70.20% → 68.50% | 64.08% → 63.68% |

Task 2 boundary 未出现 SAP 当场崩溃。Task 5 boundary 的 seen average 因 SAP 从 64.08% 降至 63.68%，变化为 -0.40 个百分点。

## SAP 诊断指标

以下四元组均按 `min / median / mean / max`，三元组均按 `min / median / max` 排列。

| Task | Feature norm before | Feature norm after | References | Gram trace |
|---|---|---|---:|---:|
| 1 | 14.4934 / 10483.1396 / 10268.4756 / 22151.5234 | 0.99999976 / 1.0 / 1.0 / 1.00000012 | 8499 | 8499.0000 |
| 2 | 12.5476 / 3294.3015 / 3219.4602 / 7075.3276 | 0.99999976 / 1.0 / 1.0 / 1.00000012 | 8494 | 8493.9990 |
| 3 | 11.9908 / 2641.2610 / 2550.5891 / 6333.5601 | 0.99999976 / 1.0 / 1.0 / 1.00000012 | 8494 | 8493.9980 |
| 4 | 14.6808 / 3411.7371 / 3335.2651 / 6604.1069 | 0.99999976 / 1.0 / 1.0 / 1.00000012 | 8496 | 8495.9990 |
| 5 | 13.6253 / 4943.7061 / 4705.2085 / 7483.1602 | 0.99999976 / 1.0 / 1.0 / 1.00000012 | 8495 | 8494.9990 |

| Task | Normalized energy | Importance | Relative weight delta | Weight norm ratio |
|---|---|---|---:|---:|
| 1 | 5.48434e-8 / 9.61908e-7 / 0.976061 | 0.000164503 / 0.00287742 / 0.999992 | 0.819206 | 0.501287 |
| 2 | 1.27910e-7 / 1.76644e-6 / 0.966866 | 0.000383584 / 0.00527139 / 0.999989 | 0.372247 | 0.866571 |
| 3 | 2.11835e-7 / 2.29680e-6 / 0.959594 | 0.000635100 / 0.00684328 / 0.999986 | 0.335292 | 0.859607 |
| 4 | 1.72685e-7 / 2.02962e-6 / 0.957235 | 0.000517788 / 0.00605202 / 0.999985 | 0.320539 | 0.857025 |
| 5 | 9.51858e-8 / 1.43278e-6 / 0.960173 | 0.000285476 / 0.00427994 / 0.999986 | 0.368138 | 0.816290 |

## 一致性检查

所有 Task boundary 均满足：

- normalization 后 feature norm 约等于 1；
- `gram_trace` 约等于 `total_reference_count`；
- 全部 oracle reference 均参与 SAP；
- SAP 事件状态均为 `SAP_ORACLE_EXECUTED`。

## 产物

- 完整标准日志：`run_logs/cifar10_symm20_linear_oracle_l2norm_alpha3000_full5_launchd_20260902/train.log`
- tqdm / stderr 日志：`run_logs/cifar10_symm20_linear_oracle_l2norm_alpha3000_full5_launchd_20260902/train.err.log`
- 最终 checkpoint：`checkpoints/dgc-sap_seq-cifar10_None_500_50_20260902-091803_53d5e804_last.pt`

训练日志与 checkpoint 是本地运行产物，未纳入 Git 提交。
