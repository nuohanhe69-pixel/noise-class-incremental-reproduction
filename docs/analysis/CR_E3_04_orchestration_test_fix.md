# CR-E3-04：E3 orchestration-level targeted tests 补齐

## 1. 修改范围

本 CR 只修改：

- `tests/test_e3_task10_gram_centering.py`

并新增本报告：

- `docs/analysis/CR_E3_04_orchestration_test_fix.md`

`scripts/evaluate_e3_task10_gram_centering.py` 未修改。训练流程、SAP 主流程和 E3 实验实现均未修改。

## 2. Synthetic orchestration fixture

测试新增一个完全 synthetic 的 source run fixture，包含：

- 固定 run id 目录；
- synthetic checkpoint 占位文件；
- `W_before.pt`；
- `W_after_taskwise.pt`；
- 已逐样本 L2 normalized 的 `X_global.pt`；
- `trusted_task_ids.pt`；
- 已保存的 `G_task_9.pt`；
- 已保存的 `M_task_9.pt`；
- source `accuracy.json`；
- 10-task dataset fake、classifier 和 checkpoint args。

真实文件读取、Gram centering、centered projection、candidate 构造、accuracy sanity、artifact 写入和 `finally` 恢复仍由 `run_e3()` 的实际实现执行。只 mock 了 checkpoint/model 初始化、test-loader 构造和昂贵的 dataset evaluation 边界。

## 3. 新增测试

### 3.1 完整成功路径

`test_run_e3_loads_source_then_evaluates_control_before_treatment_and_saves`

验证：

- checkpoint args 和 model checkpoint 均被加载；
- source tensor/artifact 由真实 loader 读取并用于计算；
- checkpoint 中的 `sap_oracle_scale=3000` 实际传入 `build_centered_projection()`；
- 第一次 evaluate 的 candidate 精确等于 checkpoint Current-Local Control；
- 第二次 evaluate 才是 Treatment；
- Treatment 的旧 rows 与 Control 精确一致；
- 两次 evaluation 后生成全部成功 artifact；
- 保存的 Control/Treatment accuracy 与 evaluator 返回值一致；
- 保存的 Task10 mean 来自 source `X_global` 中 Task10 rows。

### 3.2 Control accuracy mismatch fail-fast

`test_run_e3_control_accuracy_mismatch_stops_before_treatment_and_save`

验证 source Current-Local accuracy 不匹配时：

- `run_e3()` 抛出 `AssertionError`；
- evaluator 只调用一次，即 Treatment 不执行；
- `save_e3_artifacts()` 不调用；
- 不创建成功 output directory。

### 3.3 Evaluation 异常恢复

`test_run_e3_restores_classifier_when_treatment_evaluation_raises`

Control sanity 通过后，在 Treatment evaluation 人工抛出异常，验证：

- 异常继续向上传播；
- classifier weight 与进入 `run_e3()` 前的 snapshot bitwise equal；
- classifier bias 与进入 `run_e3()` 前的 snapshot bitwise equal；
- 不产生成功 artifact directory。

### 3.4 Artifact save 异常恢复

`test_run_e3_restores_classifier_when_artifact_save_raises`

两路 evaluation 均完成后，在 artifact save 人工抛出异常，验证：

- 异常继续向上传播；
- classifier weight 精确恢复；
- classifier bias 精确恢复；
- 不产生成功 artifact directory。

## 4. 是否改变 E3 实验逻辑

**NO**。

没有修改：

- centering 公式；
- alpha=3000；
- projection helper；
- Control / Treatment 构造；
- Task1～Task9 rows；
- inference feature；
- 训练或 SAP 主流程；
- 服务器运行方式。

## 5. 测试结果

执行：

```text
conda run -n nrgp-mammoth python -m pytest -q \
  tests/test_e3_task10_gram_centering.py
```

结果：

```text
12 passed in 1.87s
```

## 6. git diff --check

执行：

```text
git diff --check
```

结果：PASS，无 whitespace error。

## 7. 结论

CR-E3-03 的 MEDIUM-1 已通过 orchestration-level synthetic tests 补齐。测试现在实际覆盖成功路径、Control accuracy fail-fast、Treatment 禁止执行、artifact 禁止保存，以及 evaluation/save 两种异常路径下的 classifier weight/bias 恢复。

- 是否改变 E3 实验逻辑：**NO**
- 是否建议进入 commit/server 阶段：**YES**
