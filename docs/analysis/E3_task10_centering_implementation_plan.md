# E3 Final-only Task10 Gram Centering 最小实现方案

Source run：`b68ebc16-071c-41f9-8011-79bb8d4e2334`

本方案是对同一 Final-only / last-only source run 的离线 post-hoc control，不是新的训练实验。本地工作区未找到该 run 的真实 artifact 目录，因此本 CR 根据已验证的保存代码和 helper 契约完成实现定位；真实 shape、count、hash 和数值一致性必须在服务器运行前 fail-fast 验证。

## 可复用的现有文件 / helper

### 同一 source run 的必需输入

- final checkpoint：恢复冻结 backbone、classifier bias、source Current-Local classifier weight 和 checkpoint args。
- `W_before.pt`：提供同一 pre-SAP Task10 classifier block。
- `X_global.pt`：提供已由 source run 提取并保存的 normalized feature，不重新运行 backbone。
- `trusted_task_ids.pt`：用于从 `X_global` 中只选出 zero-based task id `9`。
- `G_task_9.pt`：作为 uncentered control 的已保存 Gram，只读复用，不重算 uncentered Gram。
- `M_task_9.pt`：作为 uncentered control 的已保存 projection，只读复用，不重算 uncentered M。

建议同时读取同目录的 `W_after_taskwise.pt` 和 `accuracy.json`作为 source Current-Local 的额外同源/精度 sanity，但它们不参与 centered Gram 或 M 的计算。所有输入都应记录绝对路径和 SHA-256。

### `X_global` 的语义已可确认

`models/aer_sap.py` 中的 source 路径是：

```text
features
→ normalize_classifier_input_features(features)
→ x_global
→ _save_taskwise_artifacts(..., x_global=x_global)
→ X_global.pt
```

因此 `X_global.pt` 保存的就是原实验已做过 sample-wise L2 normalization 的 classifier-input feature。E3 不需要也不允许重新提取 feature。

### 应复用的现有数学 / evaluation helper

- `utils.sap.build_sap_projection_from_gram(gram, scale)`：独立 post-hoc 脚本构造 `M_centered` 的首选 helper。它复用已验证的 Gram 校验、EVD、negative-eigenvalue clamp、SAP importance 公式和对称化，不需要在 E3 重写 importance。
- `utils.sap.project_linear_weight(weight, projection)`：用于 Task10 row block projection，保持正式实现的 `W @ M.T`。
- `utils.sap.resolve_classifier_module()`：定位 final Linear。
- `utils.checkpoints.mammoth_load_checkpoint()` 与 `main.initialize()`：按 checkpoint args 恢复原 model/dataset/backbone。
- `AerSap._install_classifier_candidate()` 与 `AerSap._summarize_evaluation()`：安装完整 classifier candidate、保持 checkpoint bias，并复用现有 `dataset.evaluate()` 的 Class-IL / Task-IL。
- E1 脚本中的 checkpoint provenance、完整 test-loader 构造和 candidate evaluation 模式可作为 E3 编排模板；不应因 E3 去修改 E1 或训练主流程。

## 最小新增文件

修改阶段建议只新增：

1. `scripts/evaluate_e3_task10_gram_centering.py`
   - 只读加载同一 source run 的 checkpoint/artifact。
   - 构造并评估 uncentered control 与 centered treatment。
   - 保存 E3 最小 artifact 和 provenance/sanity manifest。
2. `tests/test_e3_task10_gram_centering.py`
   - 用 synthetic tensor 验证 Task10 mask、centering、无二次 normalization、helper 复用、A/B row 契约与 fail-fast。

本分析文件为 `docs/analysis/E3_task10_centering_implementation_plan.md`。不需要修改 `models/aer_sap.py`、`models/dgc_sap.py`、`utils/sap.py`、dataset、buffer、backbone、training loop 或 E1 脚本。

## 数据流

### 1. 只读恢复 source run

```text
final checkpoint
→ mammoth_load_checkpoint(..., return_only_args=True)
→ 校验 seq-cifar100 / aer-sap / seed=0 / symmetric-noise 0.2 / alpha=3000
→ initialize(checkpoint_args)
→ mammoth_load_checkpoint(..., model)
→ 建立全部 10 个 test loaders，不训练、不调用 task boundary
→ clone checkpoint classifier weight/bias
```

checkpoint 中的 `sap_oracle_scale` 必须 fail-fast 验证为 `3000`，然后将该已验证值传给 projection helper；不在 E3 中引入可调 alpha。

### 2. 获取 Task10 normalized feature

```python
x_global = load("X_global.pt")
trusted_task_ids = load("trusted_task_ids.pt").long()

task10_mask = trusted_task_ids == 9
x10 = x_global[task10_mask]
```

必须先验证：

- `x_global.ndim == 2`；
- `len(trusted_task_ids) == x_global.shape[0]`；
- `x_global.shape[1] == classifier.in_features` （当前 ResNet18 应为 512）；
- `task10_mask.sum() > 0`；
- `x10` 每行 L2 norm 在严格小容差内约等于 1，用于验证加载的确是 source normalized feature。

这里只做 boolean indexing，不重新 sampling，不改变样本数量和顺序。

### 3. 只计算 treatment 的 centered geometry

```python
mu10 = x10.mean(dim=0, keepdim=True)
x10_centered = x10 - mu10
g_centered = x10_centered.transpose(0, 1) @ x10_centered

m_centered = build_sap_projection_from_gram(
    g_centered,
    scale=validated_checkpoint_alpha,  # 必须等于 3000
)
```

`x10_centered` 之后禁止再调用 `F.normalize()` 或 `normalize_classifier_input_features()`。Centering 改变行 norm 是 treatment 的一部分；如果再次 L2 normalization，就会同时改变第二个实验变量。

### 4. 保持 old rows 与 source Current-Local 一致

```text
source Current-Local full weight
= final checkpoint classifier.weight
（可再与同 run 的 W_after_taskwise.pt 做严格比对）
```

Treatment 不加载或重算 Task1~Task9 的 Gram/M/projection，而是直接 clone source Current-Local full weight，然后只替换 `dataset.get_offsets(9)` 返回的 current row block。这是保证 Task1~Task9 bitwise identical 的最窄路径。

### 5. 评估与恢复

```text
A Current-Uncentered-Local → existing dataset.evaluate()
B Current-Centered-Local   → existing dataset.evaluate()
finally → restore checkpoint classifier.weight and bias
```

两路使用同一 checkpoint backbone、bias、test loaders 和 evaluation mask。不调用 `train()`、`meta_begin_task()`、`meta_end_task()` 或任何 SAP boundary。

## Control / Treatment 构造方式

### A. Current-Uncentered-Local

Control 必须直接使用 source run 已有 Current-Local：

```python
w_uncentered = checkpoint_classifier_weight.clone()
```

不通过 `X10.T @ X10` 重算 `G_task_9`，也不再从 Gram 构造 `M_task_9`。已保存的 `G_task_9.pt` 和 `M_task_9.pt` 作为 control geometry/operator 直接加载。

为了证明 checkpoint current block 的确来自该 saved operator，可以只执行一次：

```python
control_task10_reconstructed, _ = project_linear_weight(
    w_before[current_start:current_end],
    m_task_9,
)
assert_close(
    control_task10_reconstructed,
    w_uncentered[current_start:current_end],
)
```

这只是使用已保存 `M_task_9` 重建 weight block 的 provenance check，不是重算 uncentered Gram、EVD、importance 或 M，而且 control candidate 仍使用 source checkpoint tensor。

### B. Current-Centered-Local

```python
w_centered = w_uncentered.clone()
w10_centered, _ = project_linear_weight(
    w_before[current_start:current_end],
    m_centered,
)
w_centered[current_start:current_end] = w10_centered
```

因此：

- Task1~Task9：A/B 均是 source Current-Local 中完全相同的 rows。
- Task10：A 使用 saved `M_task_9`所对应的 source rows，B 使用唯一新构造的 `M_centered`。
- A/B 不串联 projection；B 的 Task10 从同一 `W_before` Task10 block 出发，不是在 A 的已投影 Task10 rows 上再投影。
- classifier bias 始终是 checkpoint bias。

A/B 唯一变化是 Task10 projection matrix：`M_task_9` vs `M_centered`。

## Sanity checks

任意一项关键检查失败都应立即停止，不继续 A/B evaluation，不写“成功”artifact，不做机制解释。

1. **Source provenance**
   - checkpoint args 匹配 `seq-cifar100` / `aer-sap` / seed 0 / symmetric noise 0.2 / `sap_oracle_scale=3000`。
   - source artifact directory 必须是 run id `b68ebc16-071c-41f9-8011-79bb8d4e2334`；所有输入记录绝对路径和 SHA-256。
   - checkpoint classifier 若有 `W_after_taskwise.pt`，必须 exact 或严格小容差一致；control accuracy 若有 source `accuracy.json["taskwise"]`，必须逐项复现。
2. **Tensor alignment / validity**
   - `X_global` 为 finite floating `[N, 512]`；`trusted_task_ids` 为长度 N 的整数 tensor。
   - `W_before` shape 等于 checkpoint classifier weight shape，且 `G_task_9` / `M_task_9` 都是 finite `[512, 512]`。
   - Task10 mask 非空，`x10.shape[0]` 与 source coverage/task count（如有）一致。
3. **Normalized source feature**
   - centering 前 `x10.norm(dim=1)` 的 min/median/mean/max 均约等于 1。
   - 不在 centering 后对 row norm 施加 1 的约束；manifest 明确 `second_l2_normalization=false`。
4. **Control 不重算**
   - uncentered G 直接来自 `G_task_9.pt`，uncentered M 直接来自 `M_task_9.pt`。
   - 脚本中不存在从 `X10` 构造 uncentered Gram 或从 saved G 重建 uncentered M 的分支。
   - `trace(G_task_9)` 应在浮点容差内约等于 Task10 sample count，这使用 saved G 和 normalized-feature 不变量，不重算 uncentered Gram。
5. **Centering identity**
   - `x10_centered.mean(dim=0)` 的 absolute max 接近 0。
   - `G_centered` finite、symmetric、shape `[512, 512]`、trace > 0。
   - 利用 saved control Gram 验证：

     ```text
     G_task_9 - G_centered ≈ N10 * (mu10.T @ mu10)
     trace(G_centered) ≈ trace(G_task_9) - N10 * ||mu10||^2
     ```

     该检查只使用 saved `G_task_9`，不另行计算 `X10.T @ X10`。
6. **Projection helper / alpha**
   - treatment 只调用一次 `build_sap_projection_from_gram(G_centered, scale=3000)`，不重写 importance 公式。
   - `M_centered` finite、symmetric、shape `[512, 512]`。
   - control Task10 block 必须能用 saved `M_task_9` 和 `W_before` 以严格小容差重建。
7. **A/B 唯一变量**
   - current offsets 来自 `dataset.get_offsets(9)`，不硬编码 `90:100`。
   - current block 之外 `A == B` 使用 `torch.equal` 位级验证。
   - A current rows 是 source Current-Local；B current rows 严格等于 `project_linear_weight(W_before_current, M_centered)` 的输出。
   - A/B 每次 evaluation 前恢复同一 checkpoint bias；全部完成或任何异常后，`finally` 恢复 checkpoint weight/bias。
8. **Evaluation**
   - 复用现有 `dataset.evaluate()` 和 Task-IL mask，不新写 accuracy。
   - 保存 A/B overall Class-IL、overall Task-IL、per-task Class-IL 和 per-task Task-IL，全部为百分比。
   - 不自动输出机制结论，只输出事实数值、hash 和 sanity 状态。

Targeted tests 应至少覆盖：Task10 boolean mask 不改变 sample 顺序；centering 后均值近 0；saved uncentered G/M 不重算；projection helper 以 scale 3000 调用；不调用二次 normalization；A/B old rows bitwise equal；B 只替换 non-uniform `get_offsets(last_task_id)` 所指 block；同一 bias安装；sanity failure fail-fast；output directory 拒绝覆盖。

## 是否需要重新训练：YES / NO

**NO。**

source final checkpoint 已提供冻结 backbone、bias 和 Current-Local control weight；`W_before.pt` 提供唯一 pre-SAP classifier 起点；`X_global.pt` + `trusted_task_ids.pt` 可直接给出 normalized `X10`；`G_task_9.pt` + `M_task_9.pt` 提供 uncentered control。E3 只需离线计算 `mu10`、`X10_centered`、`G_centered`、`M_centered` 和 Task10 treatment weight block。

## 是否建议进入修改阶段：YES / NO

**YES。**

现有 artifact 语义和 helper 已足以支持一个独立 post-hoc E3 脚本，无需触碰训练/SAP 主流程。建议修改顺序为：先用 synthetic tensor 写失败测试固化 centering/无二次 normalization/A-B row 契约；再实现纯 tensor 构造；最后接入 checkpoint 恢复、现有 evaluation、artifact 与 `finally`。服务器真实 E3 必须在同一 run 的全部 preflight 通过后才执行 A/B evaluation。
