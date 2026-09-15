# 1. 修改摘要

在 `aer-sap` 上恢复了 Final-only timing，并在最后一个 Task 的 AER boundary 完成后执行一次 Identity / Global SAP / Task-wise SAP 三路对照。

三路共用同一次 Oracle Reference sampling、同一次 final Linear 输入 feature extraction、同一次 sample-wise L2 normalization 和同一个 `W_before`。三路评估完成后，最终写回 Task-wise 权重，并只在此时刷新一次 `past_model_ckpt`。

没有运行完整 CIFAR100 训练，也没有创建 commit。

# 2. 实际修改文件

- `models/aer_sap.py`
  - Final-only gate。
  - 三路 candidate 构造和现有 evaluation 调用。
  - Task-wise row projection、coverage 统计、artifact 保存和最终状态提交。
- `models/dgc_sap.py`
  - `_build_oracle_reference_batches()` 增加可选 `return_task_ids=False`。
  - 默认三元返回及 dgc-sap 现有执行路径保持不变。
- `tests/test_aer_sap.py`
  - Final-only timing、单次 Reference/feature、三路同源、row slicing、bias、最终权重/checkpoint 和 artifact 测试。
- `tests/test_sap_oracle_smoke.py`
  - trusted task-id 对齐、Old/New 来源和默认返回兼容测试。
- `docs/analysis/CR02_finalonly_taskwise_sap_implementation.md`
  - 本报告。

# 3. Final-only gate 实现

Gate 位于 `AerSap.end_task()`：

1. 始终先执行 `super().end_task(dataset)`，保留完整 AER boundary。
2. 当 `self.current_task != dataset.N_TASKS - 1` 时，仅记录 `SAP_SKIPPED_FINAL_ONLY` 并返回。
3. 最后一个 Task 才进入 `_run_task_boundary_sap()` 和新的三路对照逻辑。

判断使用项目 `dataset.N_TASKS`，没有硬编码 Task index 9 或 Task number 10，也没有修改 training loop。

# 4. Reference task_id 数据流

`DgcSap._build_oracle_reference_batches()` 的默认调用仍返回：

```text
all_images, all_true_labels, stats
```

仅 `aer-sap` 最终三路实验使用 `return_task_ids=True`，返回：

```text
all_images, all_true_labels, trusted_task_ids, stats
```

task id 构造规则：

- New/current：对已经由原 class-balanced sampling 选中的样本填入 `self.current_task`。
- Old/buffer：使用与 `clean & historical` 完全相同 mask 过滤后的 `buffer.task_labels`。
- 拼接顺序与原 Reference 一致：New 在前，Old 在后。

Reference builder 只调用一次。没有二次 sampling、Task/Class 再平衡或随机 generator 修改。

# 5. W_before 三路分支实现

在任何 projection 前解析 final classifier，并保存：

```python
W_before = classifier.weight.detach().clone()
bias_before = classifier.bias.detach().clone()
```

三路候选为：

- Identity：`W_before`
- Global：`project_linear_weight(W_before, M_global)`
- Task-wise：从 `W_before.clone()` 开始，每个 Task 仅覆盖对应 classifier rows

Global 和 Task-wise 均直接使用 `W_before`，没有读取前一个 candidate 的 classifier 当前值，因此不存在 projection 串联。

# 6. Global candidate

同一份 Reference 只执行一次：

```text
collect_classifier_input_features()
→ normalize_classifier_input_features()
→ X_global
```

Global 分支使用：

```text
G_global = X_global.T @ X_global
M_global = _build_oracle_projection(G_global)
W_after_global = project_linear_weight(W_before, M_global)
```

继续复用当前 `_build_oracle_projection()`、`_sap_importance_from_energy()` 和 `project_linear_weight()`，没有修改 L2 normalization、EVD、importance 或 projection 数学。运行正式协议时 `sap_oracle_scale` 必须传入固定值 3000。

# 7. Task-wise candidate

对 `task_id in range(dataset.N_TASKS)`：

```text
X_t = X_global[trusted_task_ids == task_id]
G_t = X_t.T @ X_t
M_t = _build_oracle_projection(G_t)
start_c, end_c = dataset.get_offsets(task_id)
W_after_taskwise[start_c:end_c] =
    project_linear_weight(W_before[start_c:end_c], M_t)
```

classifier rows 使用 `dataset.get_offsets()`，没有硬编码 `task_id * 10`。Task-wise 不重新运行 backbone 或 normalization。

# 8. Identity / Global / Task-wise evaluation

依次把以下 candidate copy 到完整 classifier：

1. `W_before`
2. `W_after_global`
3. `W_after_taskwise`

每次均恢复 `bias_before` 并调用现有 `dataset.evaluate(model, dataset)`。Class-IL 和 Task-IL 沿用现有完整 classifier + `mask_classes()/dataset.get_offsets()` 机制。

`accuracy.json` 中所有 accuracy 统一为百分比，包含：

- Overall Class-IL
- Overall Task-IL
- Per-task Class-IL
- Per-task Task-IL

# 9. 最终 classifier.weight / past_model_ckpt 状态

Identity 和 Global candidate 评估期间不刷新 checkpoint。

三路评估和 artifact 保存成功后：

```text
classifier.weight = W_after_taskwise
classifier.bias = bias_before
past_model_ckpt = deepcopy(net.state_dict())
```

因此 `meta_end_task()` 返回后的框架官方 evaluation 和最终 checkpoint 均对应 Task-wise SAP。

任一阶段失败时恢复 `W_before` 和 `bias_before`，不把 Global 或部分 Task-wise 状态写入 checkpoint。

# 10. 保存的 artifact

专用目录复用现有 `base_path/results_path`，完整结构为：

```text
<base_path>/<results_path>/<setting>/<dataset>/aer_sap/
  final_only_taskwise_sap/<conf_jobnum>/
```

保存文件：

- `W_before.pt`
- `X_global.pt`
- `trusted_labels.pt`
- `trusted_task_ids.pt`
- `coverage.json`
- `M_global.pt`
- `M_task_0.pt` ～ `M_task_9.pt`
- `W_after_global.pt`
- `W_after_taskwise.pt`
- `accuracy.json`

所有 tensor 均以 `detach().cpu()` 保存。日志记录最终 artifact 目录。

# 11. Missing Class / Empty Task 行为

`coverage.json` 对全部 100 个内部 class id 和全部 Task 记录：

- class trusted count
- task trusted count
- expected classes
- present classes
- missing classes
- coverage
- empty 状态

Missing Class 仅统计，不修改 Reference。

如果任一 Task Reference 完全为空：

1. 记录 `TASK_REFERENCE_EMPTY` 和对应 task id；
2. 终止 Task-wise SAP；
3. 恢复 pre-SAP classifier；
4. 记录 `SAP_FAILED`；
5. 不使用 Global M、Identity、补样本、重采样或再平衡作为 fallback。

# 12. 测试结果

执行命令：

```bash
conda run -n nrgp-mammoth python -m pytest \
  tests/test_aer_sap.py tests/test_sap_oracle_smoke.py -q
```

结果：

```text
18 passed in 2.59s
```

覆盖的关键契约：

- T1～T9 只跳过 SAP；AER boundary 每次都执行。
- T10 只进入一次 SAP。
- Reference builder 和 feature normalization 各调用一次。
- trusted task ids 与 images/labels 对齐。
- dgc-sap 默认 Reference builder 仍为三元返回。
- Identity / Global / Task-wise 均从同一 `W_before` 构造。
- Task-wise 按 classifier row 维切分，不与 Global 串联。
- 十个 `M_t` 均保存为 `[512,512]`。
- bias 不变。
- 最终 classifier 和 `past_model_ckpt` 对应 Task-wise。

# 13. 与 CR 要求是否存在偏差

无。

# 14. 尚存风险

- 本轮按要求未运行完整 CIFAR100，因此真实 Task10 Reference coverage、十一次 512 维 EVD 的服务器耗时和实际 artifact 大小仍需正式运行确认。
- `sap_oracle_scale` 的 parser 默认值保持现状；服务器实验命令必须显式传入 `--sap_oracle_scale 3000`，避免误用默认值。
- 如果真实 buffer 导致某个 historical Task 没有任何 Oracle-clean trusted sample，执行会按协议 fail-closed，不会生成伪造的 Task-wise candidate。

# 15. 是否建议进入服务器实验

YES。针对性单元测试和 Oracle smoke tests 已通过，Reference、feature、W 起点、row projection、bias、最终 checkpoint 和 artifact 契约均有覆盖；建议在代码 Review 通过后，以显式 `--sap_oracle_scale 3000` 启动服务器实验。
