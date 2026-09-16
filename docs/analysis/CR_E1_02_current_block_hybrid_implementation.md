# 1. 修改文件

本 CR 只新增以下文件：

- `scripts/evaluate_e1_current_block_hybrid.py`
- `tests/test_e1_current_block_hybrid.py`
- `docs/analysis/CR_E1_02_current_block_hybrid_implementation.md`

未修改任何已有训练、AER、SAP、dataset、buffer、backbone 或 evaluation 文件。工作区中 CR 开始前已存在的 `docs/analysis/E1_current_block_hybrid_control_implementation_plan.md` 保持未动。

# 2. 实际实现结构

独立脚本从 checkpoint 读取实验参数，通过项目现有 `initialize()` 恢复 dataset/model/backbone，通过 `mammoth_load_checkpoint()` 恢复最终模型。脚本只加载已保存的 `W_before`、`W_after_taskwise`、`M_global`、各 `M_task_t` 和 source `accuracy.json`，然后调用现有 classifier projection/evaluation helper 完成离线对照。

主要入口为 `run_e1()`，纯 tensor 构造集中在 `build_hybrid_candidates()`，sanity 校验与 artifact 保存分别由独立小函数承担。

# 3. Shared Old-Local Base 如何实现

`build_hybrid_candidates()` 先只创建一次 `old_local_base = W_before.clone()`。对 `0 .. N_TASKS-2` 的每个 task，仅调用一次现有 `project_linear_weight()`，并把该 task 的投影 row block 写入这个 base。最后一个 task 的 rows 在 base 中保持 `W_before` 原值。

Current-Local、Current-Identity、Current-Global 都从这一个已计算 base 分别 `clone()`，old-task projection 没有为三路重复计算。

# 4. A/B/C 如何保证唯一变化只发生在 Current rows

Current block 使用 `dataset.get_offsets(last_task_id)` 获取，没有硬编码 `90:100` 或每 task 的类别数。

- Current-Local：仅将 current rows 替换为 `project_linear_weight(W_current, M_task_last)`。
- Current-Identity：current rows 保留 `W_before` 原值。
- Current-Global：仅将 current rows 替换为 `project_linear_weight(W_current, M_global)`。

`assert_candidate_row_contracts()` 对 A/B/C 的所有 old rows 执行 `torch.equal` 位级检查，并对 B 的 current rows 与 `W_before` 执行 `torch.equal`。

# 5. Checkpoint / Identity 恢复逻辑

1. 使用 `mammoth_load_checkpoint(..., return_only_args=True)` 读取 source args 并做 provenance fail-fast。
2. 使用现有 `initialize()` 初始化 dataset/model/backbone，再使用 `mammoth_load_checkpoint()` 加载最终 state。
3. 循环调用 dataset 现有 loader builder 仅建立 10-task test loaders；不调用任何 train 或 task lifecycle hook。
4. 立即 clone checkpoint classifier weight/bias。
5. 安装 `W_before` 且始终安装同一 checkpoint bias，先完成 Identity evaluation 并与 source `accuracy.json["identity"]` 逐项比对。
6. 只有 Identity 通过后才构造/评估 A/B/C。
7. `finally` 中通过 `AerSap._install_classifier_candidate()` 恢复 checkpoint 原始 weight/bias。

# 6. 所有 sanity checks 的实现位置

- Check 0：`validate_source_provenance()` 校验 dataset/model/seed/noise/scale 以及 debug/future-eval 状态。
- Check 1：`run_e1()` 调用 `_assert_close()` 比对 checkpoint classifier 和 `W_after_taskwise`。
- Check 2：`evaluate_identity_sanity()` 使用真实 source JSON 数值比对四类 accuracy 字段。
- Check 3：`run_e1()` 调用 `_assert_close()` 检查 Current-Local 重建 `W_after_taskwise`，并记录 max-absolute/relative delta。
- Check 4/5：`assert_candidate_row_contracts()` 检查 old rows 位级相等和 B current rows 位级还原。
- Check 6/7：`assert_post_evaluation_contracts()` 检查 A/B/C old-task Task-IL 以及 B current-task Task-IL 与 Identity 一致。
- Identity weight diagnostics：`assert_identity_diagnostics()` 要求 delta/norm-ratio/cosine 精确为 `0/1/1`。

所有关键检查失败都抛出异常，不保存一份看似成功的 E1 结果。

# 7. Evaluation 复用方式

`evaluate_weight_candidate()` 先复用 `AerSap._install_classifier_candidate()` 安装 candidate weight 和同一 checkpoint bias，再复用 `AerSap._summarize_evaluation()`。后者直接调用项目现有 `dataset.evaluate(model, dataset)`，因此 Class-IL、Task-IL mask 和百分比语义与原 Final-only 实现相同，未新写 accuracy 计算。

# 8. Artifact 输出格式

默认输出目录为 `<source-artifact-dir>/e1_current_block_hybrid/`，存在时立即失败，不覆盖。仅写入：

- `W_current_local.pt`
- `W_current_identity.pt`
- `W_current_global.pt`
- `accuracy.json`
- `weight_diagnostics.json`
- `experiment_manifest.json`

三个 weight tensor 均在保存前 `detach().cpu()`。Manifest 记录 checkpoint/source 路径、所有必要输入的 SHA-256、checkpoint provenance、classifier/device/dtype/current offsets、无训练/无中间量重建标志、sanity 状态与 candidate 评估顺序。

# 9. Targeted test 结果

命令：

```text
conda run -n nrgp-mammoth python -m pytest tests/test_e1_current_block_hybrid.py -q
```

结果：`9 passed in 2.75s`。

覆盖内容包括：shared old-local base 只计算一次、非均匀 task offsets、A/B/C current/old row 契约、clone 隔离、synthetic task-wise 重建、Identity diagnostics、Identity fail-fast、Task-IL post-check、同一 bias 恢复、provenance、最小且不覆盖的 artifact 输出，以及脚本不含被禁止的中间量构建调用。

另外，脚本与测试已通过 `py_compile`，CLI `--help` 可正常打开。

# 10. git diff --check 结果

`git diff --check` 通过，无输出。本 CR 文件均为未跟踪新文件，因此另外通过 `py_compile` 和 targeted tests 验证它们的实际内容。

# 11. 是否修改任何已有训练/SAP 主流程文件

NO。

# 12. 是否运行真实实验

NO。未加载真实 CIFAR100 checkpoint/artifact，未训练，未产生真实 E1 accuracy，也未做机制解释。

# 13. 当前已知风险

- 当前只用 synthetic tensor/unit tests 验证；真实 checkpoint 的文件完整性、source accuracy 复现性和 runtime device 环境将由下一 CR 的 preflight 实际确认。
- 建立全部 test loaders 复用 dataset 现有 `get_data_loaders()`；它会实例化对应 loader，但脚本不调用训练或 model task-boundary lifecycle。
- 如 source checkpoint 与 artifact 不属于同一 run，或数值无法以严格容差重建，脚本会按设计 fail-fast，不会自动放宽阈值。
- Artifact 保存中途的 I/O 失败可能留下一个部分目录；为避免混淆或覆盖，后续重试前需人工审查该目录。

# 14. 是否建议进入 E1 真实运行阶段：YES / NO

YES。代码范围、targeted tests、CLI 入口和静态检查已通过；下一阶段应仅使用经 Review 确认的同一 source checkpoint/artifact 运行脚本，并保留全部 preflight 检查。
