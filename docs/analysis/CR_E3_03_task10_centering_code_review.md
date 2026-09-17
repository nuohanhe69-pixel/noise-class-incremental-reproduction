# CR-E3-03：E3 Task10 Gram Centering 实现代码审查

## 1. Review 结论

**PASS WITH ISSUES**

- BLOCKING：0
- MEDIUM：1
- LOW：0
- 建议进入 commit/server 阶段：**NO**

实现代码本身未发现违反 E3 单变量实验边界的问题：它使用既有 Final-only source run 的 checkpoint/artifact 做 post-hoc evaluation，不重新训练、不重新提取 feature、不重新构造 Reference；Control 复用保存的 Current-Local 权重、`G_task_9.pt`、`M_task_9.pt`，Treatment 只替换最后任务 classifier row block，并从原始 `W_before` 出发应用一次 centered projection。

当前不建议直接进入 commit/server，原因是关键异常恢复与 Control accuracy fail-fast 仅由人工代码审查确认，targeted tests 尚未通过 orchestration-level 测试执行这些契约。该问题不证明实现错误，但属于本 CR 明确要求检查的关键测试覆盖缺口。

## 2. Review 范围

只审查本轮新增的两个代码文件：

- `scripts/evaluate_e3_task10_gram_centering.py`
- `tests/test_e3_task10_gram_centering.py`

为确认调用语义，只只读核对了既有 `utils.sap`、`models.aer_sap` 和 E1 evaluator 的直接依赖；未把这些文件纳入本轮代码质量结论，也未修改任何实现文件。

## 3. 同一 source run 与纯 post-hoc 数据流

结论：**通过**。

- source artifact directory 必须以固定 run id `b68ebc16-071c-41f9-8011-79bb8d4e2334` 结尾，否则 fail-fast（script 30、291-295、337-344 行）。
- 入口只加载 checkpoint 与既有 artifact：`W_before.pt`、`W_after_taskwise.pt`、`X_global.pt`、`trusted_task_ids.pt`、`G_task_9.pt`、`M_task_9.pt`、`accuracy.json`（346-355 行）。
- checkpoint classifier weight 会与 `W_after_taskwise.pt` 做一致性检查（416-420 行），避免把不匹配的 checkpoint 当成 Current-Local Control。
- manifest 明确记录 `training_performed=False`、`reference_rebuilt=False`、`features_reextracted=False`（526-528 行）。
- 文件中没有训练调用、Reference builder、feature collector 或 normalization helper；测试也对相关禁用符号做了静态约束（test 233-246 行）。

因此实际数据流为：保存的 normalized `X_global` / task ids / Gram / projection / classifier artifacts → 两路 post-hoc evaluation，没有重新进入 Task1～Task10 训练流程。

## 4. Control 是否直接复用已有 Current-Local / G / M

结论：**通过**。

- Control classifier 直接取 checkpoint 当前 weight，且与保存的 `W_after_taskwise.pt` 校验（script 381-385、416-420 行）。
- `G_task_9.pt` 被直接加载为 `saved_uncentered_gram`（398-400 行），用于 centered Gram 的 mean-removal identity；没有计算 `X10.T @ X10` 作为新的 uncentered Control Gram。
- `M_task_9.pt` 被直接加载为 `saved_uncentered_projection`（401-403 行），并用于从 `W_before[current_rows]` 重建 Control Task10 block（431-437 行）。
- Control evaluation 直接安装 `checkpoint_weight`（487-494 行），没有重新构造 uncentered projection。
- manifest 将 `uncentered_gram_recomputed` 和 `uncentered_projection_recomputed` 都记录为 `False`（529-530 行）。

测试中的源码禁用检查禁止 E3 文件直接出现 EVD/SVD，projection helper 的 mock 也只覆盖 centered projection 路径（test 99-119、233-246 行）。

## 5. Treatment centering 数学与第二次 L2 normalization

结论：**通过**。

实际实现严格为：

```python
task_mask = trusted_task_ids.to(x_global.device) == int(last_task_id)
x_task = x_global[task_mask]
task_mean = x_task.mean(dim=0, keepdim=True)
centered_features = x_task - task_mean
centered_gram = centered_features.transpose(0, 1) @ centered_features
```

对应 script 99-119 行。

- centering 前先对 X10 row norm 做约 1 的 fail-fast 检查（104-110 行），确认输入是 source run 已保存的 sample-wise L2 normalized feature。
- centering 后没有任何 normalize 调用；仅记录 centered row norm（153-165 行）。
- synthetic test 明确断言 centered feature 等于直接减均值的结果，并断言 centered row norms 不再全为 1（test 40-84 行）。
- 非单位范数输入会被拒绝（test 86-97 行）。

## 6. Centering 是否只用于构造 G / M

结论：**通过**。

`centered_features` 只在 `build_task10_centered_geometry()` 内用于形成 `centered_gram`；随后只有 `centered_gram` 进入 projection helper（script 421-430 行）。评估路径仅安装候选 classifier weight，然后调用既有 dataset evaluation（487-508 行），没有修改 backbone output，也没有向 inference path 传入 mean 或 centered feature。

manifest 对此记录 `inference_feature_centering=False`（532 行）。源码禁用测试也禁止 E3 文件调用 feature collector 或 feature normalization helper（test 233-246 行）。

## 7. M_centered、alpha 与 SAP importance

结论：**通过**。

- checkpoint args 先经过既有 E1 provenance validator，随后读取 `sap_oracle_scale` 并要求精确为 3000（script 364-370 行）。
- `build_centered_projection()` 再次 fail-fast 要求 scale 为 3000（176-179 行）。
- M_centered 由既有 `build_sap_projection_from_gram(centered_gram, scale=...)` 构造（179 行），E3 文件没有复制 importance/EVD 数学。
- targeted test 用 mock 断言 canonical helper 被调用一次且 scale 为 3000，并确认 2999 被拒绝（test 99-119 行）。

## 8. Treatment weight 来源、old rows 与 task offsets

结论：**通过**。

- 当前 block offsets 来自 `dataset.get_offsets(last_task_id)`（script 226 行），没有硬编码 `90:100`。
- Treatment 先 clone 完整 Control weight，再只替换 current row block（227-232 行）。
- 替换 block 的输入明确为 `weight_before[current_start:current_end]`，不是 checkpoint 中已投影的 Current-Local Task10 block（228-231 行）。
- Task1～Task9 rows 用 `torch.equal` 做 bitwise identity fail-fast（234-246 行）。
- synthetic test 使用非均匀 task offsets，验证实际取得 `[13:15]` 而非任何硬编码 row 范围；同时验证 projection 只调用一次、输入来自 `W_before`、old rows 与 Control bitwise identical（test 121-143 行）。

## 9. Sanity checks 与 finally 恢复路径

结论：**实现通过；测试覆盖不完整**。

实现中的 fail-fast 检查：

- X10 row norm ≈ 1：script 104-110 行。
- centered mean ≈ 0：114-118 行。
- `G_unc - G_centered ≈ N * mu.T @ mu`：123-131 行。
- trace identity：132-141 行。
- saved `M_task_9` + `W_before` 重建 Control Task10：191-213、431-437 行。
- old rows bitwise identical：234-246 行。
- Control Current-Local accuracy 与 source `accuracy.json["taskwise"]` 一致：487-501 行；Treatment 只在该检查通过后评估（502-508 行）。
- evaluate 或 artifact save 期间无论何处抛出异常，`finally` 都把 snapshot 的 classifier weight 和 bias 恢复（487-558 行）。进入 `try` 以前没有安装候选 classifier，因此更早的失败也不会改变 classifier。

上述 tensor-level contract 均有对应 helper test，但 Control accuracy fail-fast 与 `finally` 恢复没有由测试实际执行，详见下一节 MEDIUM issue。

## 10. 测试覆盖评估

已覆盖：

- 10-task/last task id 布局约束。
- X10 按 task id 提取、输入 row norm、直接 centering、centered mean、Gram identity。
- 禁止第二次 L2 normalization。
- canonical centered projection helper 与 alpha=3000。
- 非硬编码 offsets、从 `W_before` 投影当前 block、old rows bitwise identity。
- saved M 对 Control Task10 block 的重建及 corruption fail-fast。
- artifact 文件集合、CPU tensor 与防覆盖行为。
- E3 source 中不存在 Reference/feature rebuild、显式 normalize、EVD/SVD、训练 boundary 调用。

未覆盖：

- 没有通过 mock/fixture 驱动 `run_e3()` 完整 orchestration。
- 没有测试 source Current-Local accuracy 不匹配时 Treatment 不会运行、artifact 不会保存。
- 没有在 Control evaluate、Treatment evaluate、artifact save 分别抛异常时断言 classifier weight/bias 被 `finally` 精确恢复。
- 没有测试 checkpoint 的 alpha/provenance 如何实际流入 centered projection；现有测试只直接调用 helper 并传入 3000。

### MEDIUM-1：关键 orchestration 与异常恢复契约缺少执行级测试

CR-E3-03 第 9、10 项把“Control accuracy 可复现”和“finally 恢复原 classifier weight/bias”列为关键约束，并要求测试真正覆盖而非只测局部 helper。目前实现代码具备这些逻辑，但 8 个测试没有调用 `run_e3()`，因此对调用顺序、accuracy fail-fast、异常恢复和 checkpoint alpha 数据流的保证主要依赖源码审查。

影响：未来若 orchestration 顺序、异常范围或 helper 接线变化，现有测试仍可能全部通过，却允许 Treatment 在无效 Control 上运行，或在失败后留下被替换的 classifier 状态。该问题不改变当前实现的数学结果，因此不列为 BLOCKING；但在服务器真实运行前应补一个完全 synthetic/mocked 的 `run_e3()` orchestration test，以及至少一个 evaluate/save failure restoration test。

## 11. 验证结果

执行：

```text
conda run -n nrgp-mammoth python -m pytest -q tests/test_e3_task10_gram_centering.py
```

结果：`8 passed in 3.01s`

执行：

```text
conda run -n nrgp-mammoth python -m py_compile \
  scripts/evaluate_e3_task10_gram_centering.py \
  tests/test_e3_task10_gram_centering.py
```

结果：PASS。

执行：

```text
git diff --check
```

结果：PASS。

未运行真实 CIFAR100 E3，未创建 commit，未 push，未修改两个被审查的代码文件。

## 12. 最终建议

- Review 结论：**PASS WITH ISSUES**
- BLOCKING：**0**
- MEDIUM：**1**
- LOW：**0**
- 建议进入 commit/server 阶段：**NO**

建议先只补齐上述 orchestration-level targeted tests，再复核并进入 commit/server 阶段；无需改变当前 E3 数学或实验逻辑。
