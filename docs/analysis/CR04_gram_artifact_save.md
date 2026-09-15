# 1. 修改内容

本 CR 只为现有 Final-only Task-wise SAP artifact 增加 Gram Matrix 保存：

- 在 `models/aer_sap.py` 的 `_run_final_taskwise_sap()` 中保留已经用于 projection 的 `global_gram` 和每个 `task_gram`。
- 将这些现有 tensor 传入 `_save_taskwise_artifacts()`。
- 在 `tests/test_aer_sap.py` 的现有 targeted test 中补充文件、shape 和数值来源断言。

没有修改 Reference、trusted sample、task id、feature extraction、L2 normalization、SAP 数学、alpha、projection、evaluation、最终权重或 checkpoint 逻辑。

# 2. 新增 artifact

沿用现有目录：

```text
final_only_taskwise_sap/<run_id>/
```

新增文件：

- `G_global.pt`
- `G_task_0.pt`
- `G_task_1.pt`
- `G_task_2.pt`
- `G_task_3.pt`
- `G_task_4.pt`
- `G_task_5.pt`
- `G_task_6.pt`
- `G_task_7.pt`
- `G_task_8.pt`
- `G_task_9.pt`

保存代码统一使用：

```python
torch.save(tensor.detach().cpu(), output_path)
```

因此保存的 Gram tensor 均已脱离 autograd graph 并位于 CPU。

# 3. Gram 的真实来源

Global Gram 仍由原 projection 流程中的同一行计算：

```python
global_gram = x_global.transpose(0, 1) @ x_global
global_projection, ... = self._build_oracle_projection(global_gram)
```

保存时直接传递这个 `global_gram`：

```python
'G_global.pt': global_gram,
```

每个 Task Gram 仍由原 Task-wise projection 循环计算：

```python
task_features = x_global[task_ids_device == task_id]
task_gram = task_features.transpose(0, 1) @ task_features
task_projection, ... = self._build_oracle_projection(task_gram)
task_grams[task_id] = task_gram
```

保存时直接遍历已保留的真实 Gram：

```python
for task_id, gram in sorted(task_grams.items()):
    torch.save(
        gram.detach().cpu(), output_directory / f'G_task_{task_id}.pt',
    )
```

`G_global.pt` 和 `G_task_*.pt` 均是本轮实际传给 `_build_oracle_projection()` 的 Gram tensor。没有为了保存而再次构造 Reference、运行 backbone、提取 feature、L2 normalize 或重新计算 Gram。

# 4. 是否改变任何实验逻辑

NO。

本 CR 仅增加 tensor 引用收集、函数参数传递和磁盘保存：

- `X_global` 构造不变。
- Global/Task-wise Gram 的计算表达式和执行位置不变。
- `M_global` / `M_task_*` 仍由同一个既有 Gram 构造。
- `W_after_global` / `W_after_taskwise` 构造不变。
- Identity → Global → Task-wise evaluate 顺序不变。
- 最终 `classifier.weight = W_after_taskwise` 不变。
- bias 恢复不变。
- `past_model_ckpt` 刷新位置和次数不变。

未修改 `models/dgc_sap.py`、buffer、dataset、training loop、optimizer/scheduler 或其他禁止文件。

# 5. 测试结果

先按 TDD 增加断言并运行单测，确认 RED：

```text
1 failed
失败原因：缺少 G_global.pt 与 G_task_0.pt ～ G_task_9.pt
```

完成最小实现后，focused test：

```text
1 passed in 2.58s
```

按 CR 要求执行：

```bash
conda run -n nrgp-mammoth python -m pytest \
  tests/test_aer_sap.py tests/test_sap_oracle_smoke.py -q
```

结果：

```text
18 passed in 1.88s
```

新增断言验证：

1. `G_global.pt` 存在。
2. `G_task_0.pt` ～ `G_task_9.pt` 全部存在。
3. 所有保存的 Gram shape 均为 `(512, 512)`。
4. `G_global.pt == X_global.T @ X_global`。
5. 对每个 task，`G_task_t.pt == X_global[trusted_task_ids == t].T @ X_global[trusted_task_ids == t]`。

# 6. `git diff --check`

执行：

```bash
git diff --check
```

结果：通过，退出码 0，无输出。

# 7. 是否建议 commit

YES。改动限定在允许的实现、测试和本报告文件，指定 targeted tests 与 `git diff --check` 均通过。

按本 CR 要求，本轮未创建 commit，等待 Review。

# 8. 是否建议进入服务器实验

YES，在本 CR 代码 Review 与后续 commit 完成后可以进入服务器实验。Gram 保存复用本轮实际 projection 输入，不改变任何计算结果；正式运行仍应沿用已冻结的实验命令与显式 `sap_alpha=3000`。
