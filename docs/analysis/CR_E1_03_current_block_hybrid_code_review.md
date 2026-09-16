# 1. Review 结论

**PASS WITH ISSUES**

实际代码满足 E1 的核心因果契约：A/B/C 的唯一变量是最后一个 task classifier row block 采用的 operator，old-task rows 由同一个只计算一次的 old-local base clone 而来。未发现 Reference、feature、Gram、EVD/SVD、M 或训练流程的隐藏重算。

发现 2 个非阻塞性 LOW 问题：缺少 `run_e1()` 编排/异常恢复的 mock integration test，artifact 写入不是目录级原子操作。它们不改变当前 A/B/C 数学路径，也不会使已通过的 preflight 结果被写成错误的完整成功结果。

# 2. 实际修改范围

Review 开始时，`git status --short` 仅显示四个未跟踪文件：

```text
?? docs/analysis/CR_E1_02_current_block_hybrid_implementation.md
?? docs/analysis/E1_current_block_hybrid_control_implementation_plan.md
?? scripts/evaluate_e1_current_block_hybrid.py
?? tests/test_e1_current_block_hybrid.py
```

其中 CR-E1-02 的真实实现范围为脚本、测试和 CR-E1-02 报告三个新文件；`E1_current_block_hybrid_control_implementation_plan.md` 是前置分析产物，本轮没有修改。没有 tracked file 处于 modified 状态，因此没有触碰现有训练/SAP/dataset/evaluation 主流程。

当前 branch 为 `SAP`，HEAD 为 `37321aa8cbb09338b20df588b0c3695ce62d0099`。本 CR 只新增本 Review Markdown。

# 3. Shared Old-Local Base Review

**通过。**

`build_hybrid_candidates()` 位于 `scripts/evaluate_e1_current_block_hybrid.py:107-140`。关键数据流是：

```python
old_local_base = weight_before.clone()
for task_id in range(last_task_id):
    start_class, end_class = offsets[task_id]
    projected, _ = project_linear_weight(
        weight_before[start_class:end_class], task_matrices[task_id],
    )
    old_local_base[start_class:end_class] = projected

candidates = {
    'current_local': old_local_base.clone(),
    'current_identity': old_local_base.clone(),
    'current_global': old_local_base.clone(),
}
```

逐项结论：

1. base 只从一次 `weight_before.clone()` 开始。
2. `range(last_task_id)` 使每个 old-task local projection 只调用一次。真实 CIFAR100 为 9 次；synthetic test 为 2 次。
3. current task 不在该 loop 内，因此 base 中保留原始 current rows。
4. A/B/C 都从同一 `old_local_base` 独立 clone。
5. 没有为 A/B/C 重复计算 old projection。

`tests/test_e1_current_block_hybrid.py:43-82` 通过 mock `project_linear_weight` 的 call count 验证这一点：3-task synthetic 场景总调用数严格为 4（2 个 old local + current local + current global），不是根据结果相同反推。

# 4. A/B/C Candidate Review

**通过。**

`scripts/evaluate_e1_current_block_hybrid.py:126-140` 中：

- A / Current-Local：old rows 继承 shared base，current rows 被替换为 `project_linear_weight(W_current, M_task_last)`。
- B / Current-Identity：只 clone shared base，current rows 从未被写入，因此保持 `W_before`。
- C / Current-Global：old rows 继承 shared base，只将 current rows 替换为 `project_linear_weight(W_current, M_global)`。

`offsets` 来自 `dataset.get_offsets(task_id)`（第 116 行），代码中没有 `90:100` 或 `task_id * 10`。C 没有对整个 classifier 使用 `M_global`，三路 candidate 之间也没有串联修改。

`assert_candidate_row_contracts()`（第 143-164 行）对 A/B/C old rows 和 B current rows 使用 `torch.equal` 执行位级契约检查。

# 5. Projection 数学路径 Review

**通过。**

E1 没有自行写 matmul。所有 old-local、current-local 和 current-global 投影都调用导入的 `utils.sap.project_linear_weight()`。该现有 helper 在 `utils/sap.py:544-586` 的实际数学是：

```python
projected = weight @ projection.to(dtype=weight.dtype).transpose(0, 1)
```

因此 E1 保留了正式实现的 `W @ M.T`，没有因 M 理论对称而改成新路径。

# 6. Hidden Recomputation Review

**通过。**

对脚本的调用链和符号搜索均未发现：

- Oracle Reference builder 或 Reference sampling；
- classifier-input feature 提取或 L2 normalization；
- Gram 构造；
- EVD/SVD；
- projection-matrix builder；
- SAP task boundary；
- `train()` / `meta_begin_task()` / `meta_end_task()`。

`run_e1()` 只从 source directory 加载 `W_before.pt`、`W_after_taskwise.pt`、`M_global.pt`、`M_task_*.pt`、`accuracy.json`，另加最终 checkpoint。导入 `AerSap` 只使用 `_install_classifier_candidate()` 和 `_summarize_evaluation()`，这两个 helper 都不进入 SAP/reference 路径。

# 7. Checkpoint / Model Restore Review

**通过。**

- `run_e1():347-350` 使用 `mammoth_load_checkpoint(..., return_only_args=True)` 读取参数并立即做 provenance 检查。
- 第 357 行将该 checkpoint args 传入项目现有 `initialize()`。
- 第 358 行再通过 `mammoth_load_checkpoint()` 把最终 state 真实加载到 model。
- 第 361-364 行在任何 classifier 替换前 clone checkpoint weight/bias。
- 第 422-426 行在 Identity 或 A/B/C 安装前，先把 checkpoint classifier 与 `W_after_taskwise` 比对。
- `evaluate_weight_candidate():261-271` 每次安装完整 candidate weight 和同一 checkpoint bias；backbone 不被写入。
- 脚本不读写 `past_model_ckpt`，不恢复 optimizer/scheduler，也不执行任何训练。

# 8. Dataset / Evaluation Review

**通过。**

`_build_all_test_loaders()`（第 324-328 行）调用 `dataset.get_data_loaders()` 正好 `dataset.N_TASKS` 次，然后要求 `len(dataset.test_loaders) == N_TASKS`。对 `SequentialCIFAR100`，dataset 初始 `c_task=-1`，每次 `store_masked_loaders()` 增加 1，10 次后 `c_task=9`；此时 `dataset.get_offsets()` 默认 end offset 为 100，现有 Class-IL evaluation 覆盖全部 100 类。

`get_data_loaders()` 会同时实例化当前 train loader，但 E1 从不迭代或训练该 loader，也不调用 model boundary。这是当前 dataset 无独立 test-loader builder 时的最小复用路径。

`validate_source_provenance()` 强制 `debug_mode=0` 和 `eval_future=False`。评估通过 `AerSap._summarize_evaluation()` 进入现有 `dataset.evaluate(model, dataset)`：Class-IL 和 Task-IL mask 没有重新实现，overall 仍是现有 Final-only JSON 相同的 per-task 百分比算术平均。

# 9. Preflight / Sanity 顺序 Review

**通过。**

`run_e1()` 的真实顺序为：

1. 第 347-350 行：checkpoint args + provenance。
2. 第 422-426 行：checkpoint weight 与 `W_after_taskwise` 比对。
3. 第 431-437 行：安装 `W_before`、Identity evaluation、与 source accuracy 逐项比对。
4. 第 439-446 行：构造 candidates，A 与 `W_after_taskwise` 重建比对。
5. 第 448-452 行：A/B/C old rows exact equality 与 B current rows exact equality。
6. 第 454-457 行：只在上述全部通过后评估 A/B/C。
7. 第 459-466 行：post-evaluation Task-IL 契约。
8. 第 505-507 行：仅在全部检查通过后保存。

不存在先评估 A/B/C 再做关键 preflight 的顺序倒置。

# 10. Floating-point Comparison Review

**通过。**

- A/B/C old rows 和 B current rows 来自相同 clone/source，使用 `torch.equal` 是合理的。
- checkpoint/saved-taskwise 与 A/saved-taskwise 先尝试 exact，否则使用 `torch.testing.assert_close(rtol=1e-6, atol=1e-7)`，并记录 max absolute/relative delta。
- Identity accuracy 及 post-evaluation accuracy 都以 percentage-point tolerance 比较，默认 `1e-4`，不使用 Python exact float equality。
- Identity diagnostics 的 exact `0/1/1` 不是对计算出的 cosine 盲目断言：`compute_weight_diagnostics()` 先用 `torch.equal(weight_before, weight_after)` 确认两个 tensor 位级相同，再直接返回数学上精确的 `delta=0, ratio=1, cosine=1`。非相同 tensor 才调用浮点 cosine 实现。

# 11. Artifact / Manifest Review

**核心契约通过，有 1 个 LOW 问题。**

`save_e1_artifacts():274-299` 保存的文件精确为：

- `W_current_local.pt`
- `W_current_identity.pt`
- `W_current_global.pt`
- `accuracy.json`
- `weight_diagnostics.json`
- `experiment_manifest.json`

`mkdir(..., exist_ok=False)` 拒绝覆盖；三个 tensor 均先 `detach().cpu()`；没有复制 X/G/M/W_before/checkpoint。`run_e1():366-383` 对 checkpoint、weight、matrix 和 source accuracy 记录绝对路径与 SHA-256。Manifest 中用以下明确布尔语义记录未执行的操作：

```text
training_performed=false
reference_rebuilt=false
features_reextracted=false
gram_recomputed=false
projection_matrices_recomputed=false
```

上述命名与需求的 `trained/features_recomputed/projection_recomputed` 文字不完全相同，但含义清楚、无反转，不影响 provenance。

所有 sanity 在第 505 行调用 save 前已通过，因此 preflight/sanity 失败不会创建输出目录。但保存过程不是目录级原子操作：I/O 在中途失败时可留下缺少完整 manifest 的部分目录。详见 Issue 2。

# 12. finally Restore Review

**通过。**

从首次 classifier 修改（Identity 安装）开始，所有修改、candidate sanity、evaluation、diagnostics、Git metadata 和 artifact 写入都在 `run_e1():421-507` 的 `try` 内。第 508-511 行的 `finally` 无条件通过现有 install helper 恢复 checkpoint weight 和 checkpoint bias。

因此 Identity mismatch、A 重建失败、row contract 失败、candidate evaluation 异常、post-check 失败和 artifact I/O 异常都会进入恢复。Provenance 失败发生在 model 初始化前；在 `try` 之前但 model 已加载的 missing/invalid input 失败也尚未修改 classifier，其状态仍是 checkpoint 原状。因此所有异常边界下都不会留下 candidate classifier 状态。

# 13. Tests Review

**核心 tensor 契约覆盖充分，有 1 个 LOW 覆盖缺口。**

1. old-local base 只计算一次：mock call count 严格检查。
2. A/B/C old rows bitwise equal：`torch.equal` 断言。
3. A current = local：synthetic scale 数值断言。
4. B current = identity：`torch.equal` 断言。
5. C current = global：synthetic scale 数值断言。
6. clone isolation：篡改 A 后检查 B/C/base 不变。
7. non-uniform offsets：使用 `(0,2), (2,5), (5,7)` 且检查 request 顺序。
8. synthetic task-wise reconstruction：A 与手工 synthetic saved tensor 比对。
9. diagnostics：Identity 的 `0/1/1` 严格断言。
10. fail-fast：Identity mismatch 时 evaluator 只被调用一次，证明 helper 不继续。
11. output-dir no overwrite：首次保存文件集精确，第二次要求 `FileExistsError`。
12-14. no Reference/feature/Gram/M rebuild：当前测试使用 module source 符号检查；人工调用链 Review 也确认无隐藏重算。
15. same bias install：在两次 evaluation 之间主动破坏 bias，并在 fake dataset 内确认每次评估看到同一 checkpoint bias。

当前测试没有通过 mock 端到端调用 `run_e1()`，因此执行顺序、八项 status 更新、candidate evaluation 失败后的 finally 恢复与“sanity 失败时不写 artifact”主要由代码级 Review 保证，而非回归测试。详见 Issue 1。

# 14. Static Checks

执行结果：

```text
conda run -n nrgp-mammoth python -m pytest tests/test_e1_current_block_hybrid.py -q
.........                                                                [100%]
9 passed in 3.01s

conda run -n nrgp-mammoth python -m py_compile \
  scripts/evaluate_e1_current_block_hybrid.py \
  tests/test_e1_current_block_hybrid.py
# PASS，无输出

git diff --check
# PASS，无输出
```

`git diff --check` 不检查未跟踪文件，因此额外对四个未跟踪文件逐个运行了 `git diff --no-index --check /dev/null <file>`；均未报 whitespace error。

# 15. Issues

## Issue 1

- **Severity：LOW**
- **文件：** `tests/test_e1_current_block_hybrid.py`
- **位置：** 全文，特别是第 108-132 行和 245-257 行
- **问题：** 测试很好地覆盖了纯 candidate/helper 契约，但没有一个 mock integration test 直接调用 `run_e1()`。Identity fail-fast 只在 helper 级验证；禁止重算主要使用 source-string 检查；没有自动验证 orchestration 中 preflight 顺序、异常后 checkpoint weight/bias 恢复、以及失败时不写 artifact。
- **对 E1 因果解释的影响：** 当前实现经逐行 Review 确认顺序和恢复正确，因此不直接污染本次 E1。但未来编排层回归时，当前 test suite 可能仍然保持绿色，降低后续变更的因果安全性。
- **建议如何修复：** 后续 CR 使用 fake checkpoint args/model/dataset/classifier 和临时 artifact，mock 加载/初始化边界，直接调用 `run_e1()`；断言关键 preflight 调用顺序、任一 sanity/evaluation 异常后 classifier 恢复，以及失败时 output directory 不存在。

## Issue 2

- **Severity：LOW**
- **文件：** `scripts/evaluate_e1_current_block_hybrid.py`
- **位置：** `save_e1_artifacts()`，第 274-299 行
- **问题：** 输出目录先直接创建，再逐文件写入。如果 disk-full、permission 变化或进程中断发生在写入中途，会留下一个部分目录，且默认 no-overwrite 会拒绝原路径重试。
- **对 E1 因果解释的影响：** 所有 sanity 在保存前已通过，且 manifest 最后写入，所以不会把 preflight 失败包装成完整成功结果。风险主要是操作人误把部分目录当成可用结果，而不是 A/B/C 数学污染。
- **建议如何修复：** 后续 CR 写入同父目录的唯一临时目录，全部成功后再原子 rename 到最终 output directory；或把一个明确 completion marker 作为最后一步。

**Blocking issues：0。**

# 16. 是否允许进入 commit/push

**YES。**

两个 LOW 问题不改变当前实验数学路径，核心契约和 fail-fast 顺序已由实际代码 Review 与 targeted tests 双重确认。

# 17. 是否允许进入服务器真实 E1

**YES。**

建议在服务器上保持当前全部 preflight，不放宽容差；只有 CLI 返回 success 且六个 artifact 齐全、manifest 可解析时，才把目录视为完整 E1 结果。
