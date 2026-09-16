# 1. 当前相关代码调用链

## 1.1 原 Final-only Task-wise SAP 训练末端调用链

当前 HEAD 为 `37321aa8cbb09338b20df588b0c3695ce62d0099`，工作树在分析开始时为 clean。

Task10 训练完成后的真实调用链如下：

```text
utils/training.py:305
model.meta_end_task(dataset)
  → models/utils/continual_model.py:546-557
    ContinualModel.meta_end_task()
      → models/aer_sap.py:509-514
        AerSap.end_task()
          → models/er_ace_aer_abs.py:119-129
            ErAceAerAbs.end_task()             # 正常 AER boundary
          → AerSap._run_task_boundary_sap()
          → models/aer_sap.py:229-493
            AerSap._run_final_taskwise_sap()
              → 构造一次 Oracle Reference
              → 提取并 L2 normalize 一次 feature，得到 X_global
              → 保存 W_before / bias_before
              → 构造 Identity / Global / Task-wise
              → 三次 dataset.evaluate()
              → 保存 artifact
              → 写回 W_after_taskwise
              → 刷新 past_model_ckpt
      → model._current_task += 1
  → utils/training.py:307
    官方 dataset.evaluate(model, dataset)
  → utils/training.py:330-335
    save_mammoth_checkpoint(...)
```

关键位置：

- `models/aer_sap.py:254-260`：在任何 projection 前快照 `classifier.weight` 和 bias。
- `models/aer_sap.py:310-320`：从 `X_global` 计算真实 `G_global`、`M_global`、`W_after_global`。
- `models/aer_sap.py:322-343`：从同一 `X_global` 按 `trusted_task_ids` 切分，并用 `dataset.get_offsets(task_id)` 投影 classifier row block。
- `models/aer_sap.py:363-378`：Identity / Global / Task-wise 三路现有评估。
- `models/aer_sap.py:397-401`：最终写回 Task-wise，并刷新 `past_model_ckpt`。
- `utils/training.py:305-335`：先执行 model boundary，再官方评估，最后保存 checkpoint。
- `utils/checkpoints.py:237-260`：checkpoint 的 `model` 字段来自当时的 `model.state_dict()`。

因此最终 checkpoint 保存的是 Task-wise candidate，而不是 Identity 或 Global。

## 1.2 建议的 E1 离线调用链

E1 不应进入 `train()` 或 `AerSap.end_task()`，建议独立脚本走以下路径：

```text
scripts/evaluate_e1_current_block_hybrid.py
  → mammoth_load_checkpoint(path, return_only_args=True)
  → main.initialize(checkpoint_args)
  → mammoth_load_checkpoint(path, model)
  → dataset.get_data_loaders() × dataset.N_TASKS
      # 只建立现有十个 test_loader；不训练、不调用 model boundary
  → resolve_classifier_module(model.net)
  → 读取 W_before.pt / W_after_taskwise.pt / M_global.pt / M_task_*.pt
  → 输入一致性与 Identity sanity checks
  → 从同一个 W_before.clone() 独立构造 A/B/C
  → 逐个安装完整 weight，bias 始终恢复为 checkpoint bias
  → dataset.evaluate(model, dataset)
  → 保存 E1 小型 artifacts
  → finally 恢复进程内原 checkpoint weight 和 bias
```

`dataset.get_data_loaders()` 会同时构造 train loader，但离线脚本不迭代 train loader、不调用 `meta_begin_task/meta_end_task`，因此不会训练，也不会构造或采样 SAP Reference。当前项目没有只构造 CIFAR100 test loaders 的更窄公共 API；复用该入口是避免修改 dataset/evaluation 的最小方案。

# 2. 当前 checkpoint / backbone / classifier / bias 状态

## 2.1 Backbone

`AerSap.end_task()` 先运行 `ErAceAerAbs.end_task()`，之后才进入 `_run_final_taskwise_sap()`。因此这里的“pre-SAP”应精确定义为：

> Task10 正常 AER boundary（包括配置中可能存在的 buffer fitting）完成后、SAP projection 开始前。

`_run_final_taskwise_sap()` 之后只通过 `_install_classifier_candidate()` 写 `classifier.weight` 和原 bias；`dataset.evaluate()` 使用 `torch.no_grad()`。没有任何代码写 backbone 参数。因此最终 checkpoint 中的 backbone 与上述时点的 pre-SAP backbone 相同。

注意：现有 artifacts 没有保存一份完整的 pre-SAP backbone，故不能用两个文件做 bitwise 对照；这个结论来自真实写路径。Identity 重评是运行时的端到端校验。

## 2.2 Classifier weight

ResNet18 的 classifier 定义在 `backbone/ResNetBlock.py:156`，为：

```python
self.classifier = nn.Linear(self.feature_dim, num_classes)
```

CIFAR100 ResNet18 下 `feature_dim=512`、`num_classes=100`，所以 weight 是 `[100, 512]`。

最终提交代码在 `models/aer_sap.py:397-401`：

```python
self._install_classifier_candidate(
    classifier, weight_after_taskwise, bias_before,
)
self.past_model_ckpt = copy.deepcopy(self.net.state_dict())
```

训练框架随后才保存最终 checkpoint。因此 checkpoint 的 `classifier.weight` 应与同一 run 的 `W_after_taskwise.pt` 一致。E1 必须在任何 evaluation 前用 `torch.testing.assert_close()` 验证；不一致时停止，防止混用 run。

## 2.3 Bias

`utils/sap.py:544-586` 的 `project_linear_weight()` 只接收 weight 和 projection，不接收 bias。`AerSap._install_classifier_candidate()`（`models/aer_sap.py:94-108`）每次都把 `bias_before` copy 回 classifier 并严格比较。

因此最终 checkpoint bias 就是 pre-SAP bias，可以直接作为 E1 全部评估的冻结 bias。现有 artifacts 没有单独的 `bias_before.pt`，但 E1 不缺该数据，因为它在最终 checkpoint 中。

## 2.4 `past_model_ckpt`

`past_model_ckpt` 是模型 Python 属性，不是注册 parameter/buffer；最终 Mammoth checkpoint 保存的是 `model.state_dict()`。E1 不依赖 `past_model_ckpt`，也不应调用 AER reload/save checkpoint 方法。

# 3. 已有 artifact 是否足够支持 E1

结论：**计算上足够，不需要重新训练或重新计算 Reference/X/G/M；但执行前必须拿到同一 run 的服务器文件并做关联校验。**

E1 实际构造只需要：

- 最终 checkpoint：backbone、bias、最终 Task-wise classifier、原实验 args。
- `W_before.pt`：A/B/C 的唯一共同起点。
- `M_global.pt`：只用于 C 的 current block。
- `M_task_0.pt` ～ `M_task_9.pt`：A 全部 block，以及 B/C 的 old blocks。
- `W_after_taskwise.pt`：Sanity Check 2 与 checkpoint/run 关联。
- 原 `accuracy.json`：Identity 与当前 Task-wise 的精确基准，不应只比较题目中四舍五入到两位小数的数字。

以下文件对 E1 candidate 构造不是必需输入，但可用于 provenance/manifest 检查，绝不能据此重算：

- `trusted_labels.pt`
- `trusted_task_ids.pt`
- `X_global.pt`
- `G_global.pt`
- `G_task_0.pt` ～ `G_task_9.pt`
- `coverage.json`

当前 artifact 目录没有把最终 checkpoint 的 hash 写入原 manifest；实际上原 CR 也没有 `experiment_manifest.json`。因此最大的运行前风险是“checkpoint 与 matrices 来自不同 run”。建议 E1 preflight 同时执行：

1. checkpoint classifier weight ≈ `W_after_taskwise.pt`；
2. 从 `W_before + M_task_0..9` 重建 A，A ≈ `W_after_taskwise.pt`；
3. checkpoint args 必须是 `dataset=seq-cifar100`、`model=aer-sap`、seed 0、symm 0.2、`sap_oracle_scale=3000`；
4. shape/dtype/finiteness 全部合法；
5. 将所有输入路径和 SHA-256 写入新的 E1 manifest。

本地工作区未发现题述 CIFAR100 artifacts、最终 checkpoint 或对应 `logs.pyd`，所以本 CR 不能直接检查真实 tensor 内容或路径。该缺失不妨碍代码方案分析，但 E1 执行必须在拥有这些服务器文件的环境进行。

# 4. 推荐的最小实现位置

推荐新增：

```text
scripts/evaluate_e1_current_block_hybrid.py
```

理由：

- E1 是 post-hoc classifier operator control，不是新训练算法。
- 不需要进入 `AerSap.end_task()`、Reference builder、feature hook、Gram/EVD 或训练循环。
- source checkpoint 和原 artifact 都可以只读。
- 项目已有独立 checkpoint 后处理脚本 `scripts/retry_sap_checkpoint.py`，说明 standalone script 符合现有组织方式。

不建议修改：

- `models/aer_sap.py`
- `models/dgc_sap.py`
- `utils/sap.py`
- `utils/evaluate.py`
- `datasets/*`
- `utils/training.py`

脚本只复用现有公共/已验证函数：

- `main.initialize()`：按 checkpoint args 创建原 model/dataset/backbone。
- `utils.checkpoints.mammoth_load_checkpoint()`：恢复完整最终 checkpoint。
- `utils.sap.resolve_classifier_module()`：定位 final Linear。
- `utils.sap.project_linear_weight()`：严格复用原 `W @ M.T` 行为。
- `AerSap._install_classifier_candidate()`：完整候选写入并保持 bias。
- `AerSap._summarize_evaluation()` 或直接调用 `dataset.evaluate()` 后使用同样字段整理结果。

建议脚本参数仅包含运行必需路径和设备：

```text
--checkpoint <final_checkpoint.pt>
--source-artifact-dir <final_only_taskwise_sap/run_id>
--output-dir <.../e1_current_block_hybrid>
--device <same device class as original, optional>
```

checkpoint 内保存的实验 args 应作为 dataset/model 配置来源，不应重新手填一套训练参数。

# 5. A/B/C candidate 的精确构造方式

必须加载而不是重新计算所有 M，并将 tensor 移到 `W_before` 的目标 device/dtype。row block 必须来自 `dataset.get_offsets(task_id)`，不能硬编码 `task_id * 10`。

原 helper 的真实公式是：

```python
projected = weight @ projection.to(dtype=weight.dtype).transpose(0, 1)
```

虽然 M 理论上对称，E1 仍必须调用 `project_linear_weight()`，从而与原 Task-wise 路径完全相同。

建议构造函数：

```python
def build_candidate(weight_before, task_matrices, current_matrix, dataset):
    candidate = weight_before.clone()
    last_task_id = int(dataset.N_TASKS) - 1
    for task_id in range(int(dataset.N_TASKS)):
        start_c, end_c = dataset.get_offsets(task_id)
        if task_id == last_task_id:
            matrix = current_matrix
        else:
            matrix = task_matrices[task_id]
        if matrix is None:
            continue
        projected, _ = project_linear_weight(
            weight_before[int(start_c):int(end_c), :], matrix,
        )
        candidate[int(start_c):int(end_c), :] = projected
    return candidate
```

三路分别独立调用：

```python
W_current_local = build_candidate(
    W_before.clone(), M_tasks, M_tasks[last_task_id], dataset,
)
W_current_identity = build_candidate(
    W_before.clone(), M_tasks, None, dataset,
)
W_current_global = build_candidate(
    W_before.clone(), M_tasks, M_global, dataset,
)
```

语义：

- A / Current-Local：Task1～Task9 用各自 `M_task_t`；Task10 用 `M_task_9`。
- B / Current-Identity：Task1～Task9 用各自 `M_task_t`；Task10 保持 `W_before[90:100]`。
- C / Current-Global：Task1～Task9 用各自 `M_task_t`；只有 Task10 用 `M_global`。

上述每次都从独立 `W_before.clone()` 开始。不能从 classifier 当前 weight 构造，也不能对 A 再改造成 B/C。

# 6. 如何复用现有 `dataset.evaluate()`

当前 `ContinualDataset.evaluate()` 在 `datasets/utils/continual_dataset.py:250-266`，转发到 `utils/evaluate.py:35-115`。

真实 evaluation 行为：

- Class-IL：`utils/evaluate.py:88-90` 在完整 seen-class logits 上取 argmax。
- Task-IL：`utils/evaluate.py:96-99` 调用 `mask_classes()`。
- `mask_classes()` 在 `utils/evaluate.py:11-25`，使用 `dataset.get_offsets(k)` 将该 task 之外的 logits 设为 `-inf`。
- 返回每 task 的 Class-IL 和 Task-IL 百分比。
- `_summarize_evaluation()` 在 `models/aer_sap.py:111-122`，对 per-task 百分比取平均，产生现有 `accuracy.json` 的四类字段。

离线脚本需要先用 checkpoint args 实例化 `SequentialCIFAR100`，再调用 `dataset.get_data_loaders()` 十次，使：

- `dataset.test_loaders` 包含十个 task；
- `dataset.c_task == 9`；
- `dataset.get_offsets()` 默认返回 `(90, 100)`，evaluation 的 `n_classes` 为 100。

每次评估应执行：

```python
AerSap._install_classifier_candidate(classifier, candidate, bias_before)
result = AerSap._summarize_evaluation(dataset, model)
```

候选建议评估顺序：

1. `identity_sanity`（`W_before`，不是 E1 的 A/B/C 之一）；
2. `current_local`；
3. `current_identity`；
4. `current_global`。

顺序本身不构成串联，因为每次都 copy 完整预构造 weight，并恢复同一个 bias。脚本应在 `finally` 中恢复从 checkpoint 刚加载时的 Task-wise weight 与 bias。

必须保留 checkpoint 中的 `debug_mode=0` 和 `eval_future=False`。`utils/evaluate.py:72` 在 debug mode 下会截断 test batches，`eval_future=True` 会走另一条 forward 逻辑；任一情况都不能用于 E1。

# 7. Identity 重建方案

恢复流程成立，精确步骤为：

1. 读取 checkpoint args，创建完全相同的 `aer-sap + seq-cifar100 + resnet18` 模型和 dataset。
2. 使用 `mammoth_load_checkpoint()` 加载最终 checkpoint；保存加载后的 `checkpoint_weight` 和 `checkpoint_bias`。
3. 构建十个现有 test loaders，不调用训练或 task boundary。
4. 从 source artifact 读取 `W_before.pt`，检查 shape `[100, 512]`、dtype、finite。
5. 将 classifier weight 完整 copy 为 `W_before`；bias 始终 copy 为 `checkpoint_bias`。
6. 调用现有 `dataset.evaluate()`。
7. 与原 `accuracy.json['identity']` 的完整精度值比较。

预期：

- Overall Class-IL 复现约 `43.75`。
- Overall Task-IL 复现约 `74.41`。
- Task10 Class-IL 复现约 `50.6`。
- Task10 Task-IL 复现约 `87.4`。

实现时不应只与上述两位/一位小数比较。应读取原 `accuracy.json`，对四个完整数组及 overall 值逐项比较。test transform 是确定性的 `ToTensor + Normalize`（`datasets/seq_cifar100.py:100-109`），同一 checkpoint、同一数据和同一 device 下应精确复现；允许一个很小的百分比容差只用于跨硬件浮点差异，例如 `abs_tol=1e-4` percentage point。任何超出容差的差异都应 fail-fast，不继续 A/B/C。

# 8. 五个 sanity checks 的具体实现方式

## Sanity Check 1：Identity accuracy

```python
identity = evaluate_candidate(W_before)
expected = source_accuracy['identity']

for key in ('class_il', 'task_il'):
    assert abs(identity[key] - expected[key]) <= 1e-4
torch.testing.assert_close(
    torch.tensor(identity['per_task_class_il']),
    torch.tensor(expected['per_task_class_il']),
    rtol=0,
    atol=1e-4,
)
torch.testing.assert_close(
    torch.tensor(identity['per_task_task_il']),
    torch.tensor(expected['per_task_task_il']),
    rtol=0,
    atol=1e-4,
)
```

失败行为：记录 manifest/preflight error，退出非零；不评估 A/B/C。

## Sanity Check 2：A 重建原 Task-wise

先把 checkpoint 最终 classifier 与保存文件关联：

```python
torch.testing.assert_close(
    checkpoint_weight.cpu(), saved_W_after_taskwise,
    rtol=0,
    atol=0,
)
```

再比较重建 A：

```python
torch.testing.assert_close(
    W_current_local.cpu(), saved_W_after_taskwise,
    rtol=1e-5,
    atol=1e-7,
)
```

第一项理论上是直接 copy/save 的同一权重，要求 exact。第二项涉及重新执行 matmul，允许小浮点误差，并在 diagnostics 记录 `max_abs_delta` 与相对 delta。失败即停止。

## Sanity Check 3：A/B/C rows 0～89 一致

不要硬编码 90；用当前 task offset 生成 old-row mask：

```python
start_c, end_c = dataset.get_offsets(dataset.N_TASKS - 1)
old_row_mask = torch.ones(W_before.shape[0], dtype=torch.bool, device=W_before.device)
old_row_mask[int(start_c):int(end_c)] = False

torch.testing.assert_close(
    W_current_local[old_row_mask], W_current_identity[old_row_mask],
    rtol=0, atol=0,
)
torch.testing.assert_close(
    W_current_local[old_row_mask], W_current_global[old_row_mask],
    rtol=0, atol=0,
)
```

这些 block 使用同一输入和同一 M，应该 bitwise identical。另验证 current rows 的来源：

```python
torch.testing.assert_close(
    W_current_identity[start_c:end_c], W_before[start_c:end_c],
    rtol=0, atol=0,
)
```

## Sanity Check 4：Task1～Task9 Task-IL 一致

```python
old_task_slice = slice(0, dataset.N_TASKS - 1)
for left, right in ((A, B), (A, C), (B, C)):
    torch.testing.assert_close(
        torch.tensor(left['per_task_task_il'][old_task_slice]),
        torch.tensor(right['per_task_task_il'][old_task_slice]),
        rtol=0,
        atol=1e-6,
    )
```

理论原因由当前 `mask_classes()` 直接保证：旧 task 的 Task-IL 会屏蔽 Task10 rows。Task1～Task9 的 Class-IL 不要求相同，因为 Class-IL 使用完整 100-class logits，Task10 block 改变可能改变旧样本的跨 task 竞争。

## Sanity Check 5：B 的 Task10 Task-IL 复现 Identity

```python
last = dataset.N_TASKS - 1
assert abs(
    current_identity['per_task_task_il'][last]
    - identity['per_task_task_il'][last]
) <= 1e-6
```

B 的 current block weight 与 Identity 完全相同、bias 相同，而 Task-IL 屏蔽所有 old rows，所以两者 Task10 Task-IL 应一致。B 的 Task10 Class-IL 不保证等于 Identity，因为 B 的 Task1～Task9 rows 已被 local projection，仍参与 Class-IL 的 100-class 竞争。

# 9. Artifact 输出设计

建议输出目录：

```text
<source-artifact-dir>/e1_current_block_hybrid/
├── W_current_local.pt
├── W_current_identity.pt
├── W_current_global.pt
├── accuracy.json
├── weight_diagnostics.json
└── experiment_manifest.json
```

不复制 `W_before`、M、G、X 或 checkpoint。

三个 weight tensor 必须以 `detach().cpu()` 保存；不得保存仍关联计算图或驻留 GPU 的 tensor。

## `accuracy.json`

统一保存百分比：

```json
{
  "identity_sanity": {
    "class_il": 43.75,
    "task_il": 74.41,
    "per_task_class_il": [],
    "per_task_task_il": []
  },
  "current_local": {},
  "current_identity": {},
  "current_global": {}
}
```

结果读取顺序固定为：Task10 Task-IL → Task10 Class-IL → Task1～Task9 Class-IL → Overall Class-IL/Task-IL。因果解释只比较这三路冻结对照：Current-Local 差而 Identity/Global 均恢复，支持 local `M_task_9` 是主要故障点；Identity 恢复但 Global 仍差，支持 Task10 对 SAP energy-based projection 普遍敏感；Global 仅部分恢复，则说明 local geometry 与 projection criterion 都可能有贡献。

## `weight_diagnostics.json`

只对 current/Task10 block 记录三路：

```python
before = W_before[start_c:end_c].flatten()
after = candidate[start_c:end_c].flatten()
denom = before.norm().clamp_min(torch.finfo(before.dtype).eps)
relative_weight_delta = ((after - before).norm() / denom).item()
weight_norm_ratio = (after.norm() / denom).item()
cosine = torch.nn.functional.cosine_similarity(before, after, dim=0).item()
```

Identity 应通过额外断言满足：

- `relative_weight_delta == 0`
- `weight_norm_ratio == 1`
- `cosine == 1`

JSON 还应记录 Sanity Check 2 的 `max_abs_delta` 和 relative reconstruction delta。

## `experiment_manifest.json`

至少记录：

- E1 名称与 git commit。
- source checkpoint/artifact 绝对路径和 SHA-256。
- checkpoint 中的 dataset/model/backbone/noise/seed/alpha。
- device、dtype、shape。
- current task id 与 `dataset.get_offsets()`。
- 输入文件清单。
- 五项 sanity check 的 pass/fail 与容差。
- 评估顺序。
- 明确声明未训练、未采样 Reference、未计算 X/G/M。

输出目录应默认拒绝覆盖已有结果，避免两次 E1 混写。

# 10. 最小需要修改/新增哪些文件

下一阶段最小范围为两个新文件：

1. `scripts/evaluate_e1_current_block_hybrid.py`
   - checkpoint/dataset/model 恢复；
   - source artifact 校验；
   - A/B/C 独立构造；
   - 复用现有 evaluation；
   - sanity checks；
   - E1 artifact 保存。

2. `tests/test_e1_current_block_hybrid.py`
   - 用小型 synthetic `[100,512]` weight 和已保存 M 测试三路构造；
   - 证明 old rows exact 相同、current rows来源正确、A 能重建保存的 Task-wise；
   - 用 fake dataset/model 验证每次评估前完整安装 candidate 和同一 bias；
   - 验证失败 sanity check 会在 candidate evaluation 前终止；
   - 验证只读取 M，不调用 Reference/feature/Gram/projection builder。

不需要修改任何现有 model、dataset、training、buffer 或 SAP 主流程文件。实现任务依赖顺序：先写纯 candidate/preflight 单测，再实现 script helper；之后补 evaluation/orchestration test；最后在服务器用真实 artifact 先只跑 Identity preflight，通过后才允许完整 A/B/C。

# 11. 风险点

只列可能污染 E1 单变量结论的真实风险：

1. **跨 run 文件混用。** 原 artifact 没有 checkpoint hash；必须用 checkpoint weight≈`W_after_taskwise`、A 重建和 SHA-256 manifest 三重约束。
2. **使用当前默认参数而非 checkpoint args。** dataset class order、base path、backbone 或 evaluation flag 漂移会导致 Identity 不可复现。必须以 checkpoint args 为源。
3. **test loaders 未完整建立。** 少于十次 `get_data_loaders()` 会导致 Class-IL seen-class 范围和 per-task 数量错误。
4. **误入训练/boundary。** 脚本禁止调用 `train()`、`meta_begin_task()`、`meta_end_task()`；否则可能触发 AER/SAP 生命周期。
5. **debug/eval_future 漂移。** `debug_mode != 0` 会截断 evaluation；`eval_future=True` 会改变 forward 路径。必须 fail-fast。
6. **候选串联。** 必须预先从同一个 `W_before.clone()` 构造完整 A/B/C，每次 evaluate 前覆盖完整 classifier weight。
7. **矩阵方向写错。** 不能手写 `W @ M`；必须复用现有 `project_linear_weight()` 的 `W @ M.T`。
8. **跨设备 matmul 微差。** A 重建可能因 CPU/GPU kernel 出现末位差异；用保存权重做小容差校验并记录误差，但不能放宽到掩盖错误的程度。
9. **错误解读旧任务 Class-IL。** A/B/C 的旧任务 Task-IL 应一致，但旧任务 Class-IL可以因 Task10 logits 改变而变化，不能把该变化自动判为实现错误。
10. **本地缺少真实 artifact。** 本 CR 未能验证题述服务器文件的内容、路径、shape 或 hash；进入真实 E1 前必须先运行只读 preflight。

# 12. 是否需要重新训练：YES / NO，并解释

**NO。**

最终 checkpoint 提供冻结的 backbone 和 bias；`W_before.pt` 提供共同 pre-SAP classifier 起点；`M_global.pt` 与 `M_task_0.pt`～`M_task_9.pt` 提供全部固定 operator。A/B/C 都是离线 classifier weight 重组，现有 `dataset.evaluate()` 可直接评估。Reference、feature、L2、Gram 和 M 均不需要也不允许重新生成。

# 13. 是否建议进入修改阶段：YES / NO

**YES。**

现有代码路径支持完全独立的 post-hoc script，最小实现无需修改训练/SAP 主流程。建议修改阶段严格限制为新增 `scripts/evaluate_e1_current_block_hybrid.py` 和其 targeted test；真实服务器运行前先完成 checkpoint/artifact preflight 与 Identity 复现，任一 sanity check 失败立即停止，不解释实验机制、不继续 A/B/C。
