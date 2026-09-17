# CR-E3-02 Task10 Gram Centering 实现报告

## 1. 修改文件

本 CR 只新增：

- `scripts/evaluate_e3_task10_gram_centering.py`
- `tests/test_e3_task10_gram_centering.py`
- `docs/analysis/CR_E3_02_task10_centering_implementation.md`

前置方案 `docs/analysis/E3_task10_centering_implementation_plan.md` 保持不变。未修改任何已有 model、SAP、training、dataset、buffer、backbone、evaluation 或 E1 文件。

## 2. 离线数据流

脚本强制 source artifact directory 的最后一级为 run id `b68ebc16-071c-41f9-8011-79bb8d4e2334`，并只读加载：

- final checkpoint
- `W_before.pt`
- `W_after_taskwise.pt`
- `X_global.pt`
- `trusted_task_ids.pt`
- `G_task_9.pt`
- `M_task_9.pt`
- source `accuracy.json`

所有输入在 manifest 中记录绝对路径和 SHA-256。Checkpoint args 必须通过已有 E1 provenance validator，且 `sap_oracle_scale` 再次严格验证为 `3000`。Dataset 必须满足 `N_TASKS == 10` 且 zero-based `last_task_id == 9`。

Task10 feature 的实际路径是：

```text
X_global.pt + trusted_task_ids.pt
→ X10 = X_global[trusted_task_ids == 9]
→ 验证 X10 每行 norm 约为 1
→ mu10 = mean(X10, dim=0, keepdim=True)
→ X10_centered = X10 - mu10
→ G_centered = X10_centered.T @ X10_centered
→ build_sap_projection_from_gram(G_centered, scale=3000)
→ M_centered
```

脚本不包含任何 feature extraction、Reference builder、uncentered Gram/M builder、EVD/SVD 重实现或二次 row normalization。Centering 只出现在 saved feature 到 treatment Gram 的离线路径，evaluation forward 不做 centering。

## 3. Control 构造

`Current-Uncentered-Local` 直接使用 final checkpoint classifier weight，并与同 run 的 `W_after_taskwise.pt` 做 exact-first / strict-tolerance 校验。

Control 不计算 `X10.T @ X10`，不从 Gram 重建 M。`G_task_9.pt` 作为 saved uncentered geometry 直接参与 centering identity sanity；`M_task_9.pt` 作为 saved uncentered operator，用于验证：

```text
project_linear_weight(W10_before, M_task_9)
≈ checkpoint Current-Local Task10 rows
```

Control evaluation 必须逐项复现 source `accuracy.json["taskwise"]`，否则 Treatment 不会被评估或保存。

## 4. Treatment 构造

Treatment 首先 clone 完整 checkpoint Current-Local weight，再仅替换 `dataset.get_offsets(9)` 返回的 current row block：

```text
W_centered = W_current_local.clone()
W10_centered = project_linear_weight(W10_before, M_centered)
W_centered[current_offsets] = W10_centered
```

因此 Task1~Task9 rows 直接来自 source Current-Local，并使用 `torch.equal` 做 bitwise equality 检查。Task10 始终从 `W_before` 出发，不在已投影的 control Task10 weight 上串联投影。Control/Treatment 始终恢复同一 checkpoint bias。

## 5. SAP 数学复用

`M_centered` 只通过现有 `utils.sap.build_sap_projection_from_gram()` 构造，该 helper 复用原有 Gram validation、EVD、energy clamp、SAP importance 和 projection 对称化。E3 没有重写 importance 公式。

Classifier projection 复用 `utils.sap.project_linear_weight()`，保持现有 `W @ M.T` 实现。

## 6. Sanity checks

保存任何成功 artifact 前实施：

1. source run id、checkpoint args、alpha、N_TASKS/last task id 检查；
2. checkpoint classifier 与 `W_after_taskwise` 一致；
3. `X_global` / task ids / classifier / saved G/M 的 shape、dtype、finite 和 alignment 检查；
4. X10 centering 前 row norm 约为 1；
5. centered feature 每维均值接近 0；
6. `G_unc - G_centered ≈ N * mu.T @ mu`；
7. `trace(G_centered) ≈ trace(G_unc) - N * ||mu||^2`；
8. centered Gram / projection finite、symmetric、shape 正确；
9. saved `M_task_9` 可从 `W10_before` 重建 control Task10 rows；
10. Control/Treatment old rows bitwise identical；
11. source Current-Local 四类 accuracy 字段逐项复现。

任意检查失败都抛出异常，不写成功目录。从首次 evaluation candidate 安装开始的所有路径都在 `try/finally` 内，无论正常、accuracy mismatch、evaluation 异常或 artifact I/O 异常，都恢复 checkpoint classifier weight/bias。

## 7. Evaluation 与 artifact

Evaluation 复用 E1 已验证的 candidate installer 和 `AerSap._summarize_evaluation()`，最终仍由现有 `dataset.evaluate()` 计算 Class-IL / Task-IL，未新写 accuracy 或 mask。评估顺序为：

1. `current_uncentered_local`；
2. source accuracy sanity 通过；
3. `current_centered_local`。

默认输出目录为 `<source-artifact-dir>/e3_task10_gram_centering/`，存在时 fail-fast，只保存：

- `task10_mean.pt`
- `G_task10_centered.pt`
- `M_task10_centered.pt`
- `W_current_centered.pt`
- `accuracy.json`
- `gram_diagnostics.json`
- `projection_diagnostics.json`
- `weight_diagnostics.json`
- `experiment_manifest.json`

所有 tensor 在保存前执行 `detach().cpu()`。Manifest 明确记录未训练、未重建 Reference、未重提 feature、未重算 uncentered G/M、未做二次 L2 和未对 inference feature centering。

## 8. Targeted tests

命令：

```text
conda run -n nrgp-mammoth python -m pytest tests/test_e3_task10_gram_centering.py -q
```

结果：`8 passed in 2.03s`。

Synthetic tests 覆盖：

- 必须是 10 tasks 且 last id 为 9；
- Task10 boolean mask 保持 source row 顺序；
- centering 直接等于 `X10 - mean`，无二次 L2；
- unit-norm source preflight 失败会拒绝；
- centered mean、Gram mean-removal identity 和 trace identity；
- canonical projection helper 只被调用一次且 scale 必须为 3000；
- Treatment 从 `W_before` current block 出发，old rows 位级不变，非均匀 offsets 不被硬编码；
- saved control projection 可重建 control Task10，错误 control 会 fail-fast；
- artifact 文件集精确、tensor 在 CPU、目录不覆盖；
- 脚本不包含 Reference/feature rebuild、二次 normalization、手写 EVD/SVD 或 task lifecycle 调用。

额外检查：

```text
python -m py_compile scripts/evaluate_e3_task10_gram_centering.py \
  tests/test_e3_task10_gram_centering.py
# PASS

python scripts/evaluate_e3_task10_gram_centering.py --help
# PASS
```

## 9. git diff --check

`git diff --check` 通过，无 whitespace error。由于本 CR 文件为未跟踪新文件，最终还对新文件运行 no-index whitespace check。

## 10. 范围结论

- 是否修改已有训练/SAP 主流程：**NO**
- 是否重新训练：**NO**
- 是否重新构造 Reference / 提取 feature：**NO**
- 是否重算 uncentered G/M：**NO**
- 是否运行真实 CIFAR100 E3：**NO**
- 是否建议进入服务器 E3：**YES**，但必须使用 run `b68ebc16-071c-41f9-8011-79bb8d4e2334` 的同源 checkpoint/artifact，并保留全部 fail-fast sanity。
