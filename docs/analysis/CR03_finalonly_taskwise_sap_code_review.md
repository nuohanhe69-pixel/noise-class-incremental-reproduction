# 1. 当前 git 状态

审查基线：

```text
branch: SAP
HEAD: dae1dca511c8756633a22a5127a4d7fd4c7503cf
HEAD subject: Revert "feat: run oracle SAP only at final task"
```

生成本报告后，`git status --short --branch --untracked-files=all` 应显示：

```text
## SAP...github/SAP
 M models/aer_sap.py
 M models/dgc_sap.py
 M tests/test_aer_sap.py
 M tests/test_sap_oracle_smoke.py
?? docs/analysis/CR02_finalonly_taskwise_sap_implementation.md
?? docs/analysis/CR03_finalonly_taskwise_sap_code_review.md
```

本次 CR-2 的实现 diff 只涉及以下四个 tracked 文件：

- `models/aer_sap.py`
- `models/dgc_sap.py`
- `tests/test_aer_sap.py`
- `tests/test_sap_oracle_smoke.py`

两个 `docs/analysis/*.md` 均为未跟踪的新文档，不在 `git diff` 的 tracked-file 统计中。

# 2. 完整 diff 摘要

`git diff --stat`：

```text
 models/aer_sap.py              | 428 ++++++++++++++++++++++++++++++++++++++++--
 models/dgc_sap.py              |  11 +-
 tests/test_aer_sap.py          | 176 +++++++++++++++++
 tests/test_sap_oracle_smoke.py |  41 ++++
 4 files changed, 654 insertions(+), 2 deletions(-)
```

## `models/aer_sap.py`

新增常量：

- `SAP_SKIPPED_FINAL_ONLY`
- `TASKWISE_SAP_STARTED`
- `TASK_REFERENCE_EMPTY`

新增函数：

- `_weight_stats()`：计算 Task-wise 完整权重相对变化与范数比例。
- `_install_classifier_candidate()`：将某一路候选权重写入同一个 final Linear，并恢复/校验原 bias。
- `_summarize_evaluation()`：复用 `dataset.evaluate()`，整理 overall/per-task Class-IL 与 Task-IL 百分比。
- `_build_reference_coverage()`：统计 class/task trusted count、每个 task 的 expected/present/missing classes、coverage 和 empty task。
- `_taskwise_artifact_directory()`：构造本实验专用 artifact 目录。
- `_save_taskwise_artifacts()`：保存共同输入、三路权重、Global/Task-wise 投影和 JSON 指标。
- `_run_final_taskwise_sap()`：完成单次 Reference、单次 feature/L2、Identity/Global/Task-wise 三路构造、三次评估、artifact 保存和最终 Task-wise 提交。

修改函数：

- `_run_task_boundary_sap()`：保留 checkpoint reconstruction/inference gate，但正常路径由旧的单路 `_run_oracle_classifier_sap()` 改为 `_run_final_taskwise_sap()`。
- `end_task()`：始终先运行 AER `super().end_task()`；仅最后一个 task 进入 SAP，前面的 task 记录 `SAP_SKIPPED_FINAL_ONLY`。

修改原因：CR-2 要求仅在最终 boundary 执行一次，并在同一 `W_before`、同一 Reference 和同一 `X_global` 上完成三路严格对照。

## `models/dgc_sap.py`

修改函数：

- `_build_oracle_reference_batches(dataset, *, return_task_ids=False)`：增加可选返回 `trusted_task_ids`。

无新增函数。具体改动是：

- New 样本以 `self.current_task` 生成 task id。
- Old 样本的 task id 使用与 `clean & historical` 相同 mask 过滤后的 `buffer.task_labels`。
- task ids 按原 Reference 的 New→Old 顺序拼接。
- `return_task_ids=False` 时继续返回原三元组，保持 dgc-sap 调用契约不变。

修改原因：Task-wise 只能按现有 Reference 的来源 task 分组，且禁止第二次 Reference 构造或重新采样。

## `tests/test_aer_sap.py`

新增测试函数：

- `test_only_final_task_runs_taskwise_sap_after_every_aer_boundary()`
- `test_final_taskwise_sap_uses_one_reference_and_one_feature_matrix()`

修改原因：覆盖 Final-only 时机、每个 task 的正常 AER boundary、单次 Reference/feature/L2、同源三路、row-wise Task projection、bias、最终 Task-wise 状态、checkpoint 和 artifacts。

## `tests/test_sap_oracle_smoke.py`

新增测试函数：

- `test_optional_task_ids_are_aligned_without_resampling_reference()`
- `test_reference_builder_default_return_contract_is_unchanged()`

修改原因：验证可选 task-id 返回与 images/labels 对齐，New/Old task id 来源正确，不产生第二轮 class sampling，并保证 dgc-sap 原三元返回不变。

# 3. Final-only `end_task` 完整实现

`AerSap.end_task()`、直接经过的 boundary gate，以及新的 Task-wise orchestration 当前完整代码如下：

```python
def end_task(self, dataset):
    super().end_task(dataset)
    if self.current_task != int(dataset.N_TASKS) - 1:
        self._record_sap_event(status=SAP_SKIPPED_FINAL_ONLY)
        return
    self._run_task_boundary_sap(dataset)
```

```python
def _run_task_boundary_sap(self, dataset) -> None:
    start_from = getattr(self.args, 'start_from', None)
    if (
        getattr(self.args, 'loadcheck', None) is not None
        and start_from is not None
        and self.current_task < start_from
    ):
        self._record_sap_event(status=SAP_SKIPPED_CHECKPOINT_RECONSTRUCTION)
        return
    if getattr(self.args, 'inference_only', False):
        self._record_sap_event(status=SAP_SKIPPED_INFERENCE)
        return
    self._run_final_taskwise_sap(dataset)
```

```python
def _run_final_taskwise_sap(self, dataset) -> None:
    """Evaluate Identity/Global/Task-wise SAP from one final-task snapshot."""
    self._record_sap_event(status=TASKWISE_SAP_STARTED)
    classifier = None
    weight_before = None
    bias_before = None
    stage = 'reference_construction'
    try:
        (
            all_images,
            trusted_labels,
            trusted_task_ids,
            reference_stats,
        ) = self._build_oracle_reference_batches(
            dataset, return_task_ids=True,
        )
        total_images = int(all_images.shape[0])
        if total_images == 0:
            raise ValueError('final task-wise SAP reference set is empty')
        if not (
            len(trusted_labels) == total_images
            and len(trusted_task_ids) == total_images
        ):
            raise ValueError('trusted images, labels, and task ids are not aligned')

        stage = 'classifier_snapshot'
        classifier = resolve_classifier_module(self.net)
        weight_before = classifier.weight.detach().clone()
        bias_before = (
            classifier.bias.detach().clone()
            if classifier.bias is not None else None
        )

        stage = 'feature_collection'
        batches_with_labels = list(
            self._normalized_batches(all_images, trusted_labels),
        )

        def image_batches():
            for images, _labels in batches_with_labels:
                yield images

        features = collect_classifier_input_features(
            self.net, image_batches(), total_images=total_images,
        )
        x_global, feature_norm_stats = normalize_classifier_input_features(features)
        if x_global.shape[0] != total_images:
            raise ValueError('normalized features are not aligned with the reference set')
        if x_global.shape[1] != weight_before.shape[1]:
            raise ValueError(
                'classifier input dimension does not match normalized features'
            )
        if weight_before.shape[0] != int(dataset.N_CLASSES):
            raise ValueError('classifier row count does not match dataset classes')

        stage = 'coverage'
        coverage = self._build_reference_coverage(
            dataset, trusted_labels, trusted_task_ids,
        )
        logging.info(
            'Task-wise SAP reference: total=%d task_counts=%s missing_classes=%s '
            'empty_tasks=%s',
            total_images,
            coverage['task_counts'],
            {
                task_id: task_stats['missing_classes']
                for task_id, task_stats in coverage['tasks'].items()
                if task_stats['missing_classes']
            },
            coverage['empty_tasks'],
        )
        if coverage['empty_tasks']:
            for task_id in coverage['empty_tasks']:
                self._record_sap_event(
                    status=TASK_REFERENCE_EMPTY,
                    reference_task_id=int(task_id),
                )
            raise ValueError(
                f"empty trusted task references: {coverage['empty_tasks']}"
            )

        stage = 'global_projection'
        global_gram = x_global.transpose(0, 1) @ x_global
        (
            global_projection,
            global_energy,
            global_normalized_energy,
            global_importance,
        ) = self._build_oracle_projection(global_gram)
        weight_after_global, global_weight_stats = project_linear_weight(
            weight_before, global_projection,
        )

        stage = 'taskwise_projection'
        weight_after_taskwise = weight_before.clone()
        task_projections = {}
        task_projection_stats = {}
        task_ids_device = trusted_task_ids.to(x_global.device)
        for task_id in range(int(dataset.N_TASKS)):
            task_features = x_global[task_ids_device == task_id]
            task_gram = task_features.transpose(0, 1) @ task_features
            (
                task_projection,
                task_energy,
                task_normalized_energy,
                task_importance,
            ) = self._build_oracle_projection(task_gram)
            start_c, end_c = dataset.get_offsets(task_id)
            task_weight, task_weight_stats = project_linear_weight(
                weight_before[int(start_c):int(end_c), :], task_projection,
            )
            weight_after_taskwise[int(start_c):int(end_c), :] = task_weight
            task_projections[task_id] = task_projection
            task_projection_stats[str(task_id)] = {
                'reference_count': int(task_features.shape[0]),
                'gram_trace': task_gram.trace().item(),
                'normalized_energy_min': task_normalized_energy.min().item(),
                'normalized_energy_median': task_normalized_energy.median().item(),
                'normalized_energy_max': task_normalized_energy.max().item(),
                'importance_min': task_importance.min().item(),
                'importance_median': task_importance.median().item(),
                'importance_max': task_importance.max().item(),
                'relative_weight_delta': task_weight_stats['relative_weight_delta'],
                'weight_norm_ratio': task_weight_stats['weight_norm_ratio'],
                'projection_shape': list(task_projection.shape),
                'gram_eigenvalue_min': task_energy.min().item(),
                'gram_eigenvalue_max': task_energy.max().item(),
            }
        taskwise_weight_stats = self._weight_stats(
            weight_before, weight_after_taskwise,
        )

        stage = 'candidate_evaluation'
        candidates = {
            'identity': weight_before,
            'global': weight_after_global,
            'taskwise': weight_after_taskwise,
        }
        accuracy = {}
        for candidate_name, candidate_weight in candidates.items():
            self._install_classifier_candidate(
                classifier, candidate_weight, bias_before,
            )
            accuracy[candidate_name] = self._summarize_evaluation(dataset, self)
            logging.info(
                'Task-wise SAP candidate %s results: %s',
                candidate_name, accuracy[candidate_name],
            )

        stage = 'artifact_save'
        output_directory = self._save_taskwise_artifacts(
            dataset,
            weight_before=weight_before,
            x_global=x_global,
            trusted_labels=trusted_labels,
            trusted_task_ids=trusted_task_ids,
            coverage=coverage,
            global_projection=global_projection,
            task_projections=task_projections,
            weight_after_global=weight_after_global,
            weight_after_taskwise=weight_after_taskwise,
            accuracy=accuracy,
        )

        stage = 'taskwise_commit'
        self._install_classifier_candidate(
            classifier, weight_after_taskwise, bias_before,
        )
        self.past_model_ckpt = copy.deepcopy(self.net.state_dict())
        logging.info(
            'Task-wise SAP final selected candidate=taskwise artifacts=%s',
            output_directory,
        )

        task_accuracy_comparisons = []
        for task_id, (before, after) in enumerate(zip(
            accuracy['identity']['per_task_class_il'],
            accuracy['taskwise']['per_task_class_il'],
        )):
            test_loader = dataset.test_loaders[task_id] \
                if task_id < len(dataset.test_loaders) else None
            sample_count = None
            if test_loader is not None and hasattr(test_loader, 'dataset'):
                sample_count = len(test_loader.dataset)
            task_accuracy_comparisons.append({
                'task_id': task_id,
                'sample_count': sample_count,
                'accuracy_before': before / 100.0,
                'accuracy_after': after / 100.0,
                'accuracy_delta': (after - before) / 100.0,
            })
        max_bias_delta = (
            0.0 if bias_before is None
            else (classifier.bias.detach() - bias_before).abs().max().item()
        )
        self._record_sap_event(
            status=SAP_ORACLE_EXECUTED,
            projection_location='pre',
            projection_target='classifier',
            projection_scope='taskwise',
            total_reference_count=total_images,
            current_task_clean_count=reference_stats['current_task_clean_count'],
            buffer_clean_count=reference_stats['buffer_clean_count'],
            current_task_clean_total=reference_stats['current_task_clean_total'],
            current_task_clean_selected=reference_stats['current_task_clean_selected'],
            historical_buffer_clean_count=reference_stats['historical_buffer_clean_count'],
            reference_new_count=reference_stats['reference_new_count'],
            reference_old_count=reference_stats['reference_old_count'],
            current_class_selected_counts=reference_stats['current_class_selected_counts'],
            reference_sampling_seed=reference_stats['reference_sampling_seed'],
            buffer_total_count=reference_stats['buffer_total_count'],
            gram_shape=list(global_gram.shape),
            gram_trace=global_gram.trace().item(),
            feature_norm_before_min=feature_norm_stats['before']['min'],
            feature_norm_before_median=feature_norm_stats['before']['median'],
            feature_norm_before_mean=feature_norm_stats['before']['mean'],
            feature_norm_before_max=feature_norm_stats['before']['max'],
            feature_norm_after_min=feature_norm_stats['after']['min'],
            feature_norm_after_median=feature_norm_stats['after']['median'],
            feature_norm_after_mean=feature_norm_stats['after']['mean'],
            feature_norm_after_max=feature_norm_stats['after']['max'],
            gram_eigenvalue_min=global_energy.min().item(),
            gram_eigenvalue_max=global_energy.max().item(),
            normalized_energy_min=global_normalized_energy.min().item(),
            normalized_energy_median=global_normalized_energy.median().item(),
            normalized_energy_max=global_normalized_energy.max().item(),
            sap_alpha=float(self.args.sap_oracle_scale),
            importance_min=global_importance.min().item(),
            importance_median=global_importance.median().item(),
            importance_max=global_importance.max().item(),
            importance_trace=global_importance.sum().item(),
            n_seen_classes=int(self.n_seen_classes),
            total_classifier_rows=int(classifier.weight.shape[0]),
            relative_weight_delta=taskwise_weight_stats['relative_weight_delta'],
            weight_norm_ratio=taskwise_weight_stats['weight_norm_ratio'],
            global_relative_weight_delta=global_weight_stats['relative_weight_delta'],
            global_weight_norm_ratio=global_weight_stats['weight_norm_ratio'],
            max_bias_delta=max_bias_delta,
            seen_task_accuracy_comparisons=task_accuracy_comparisons,
            seen_average_accuracy_before=accuracy['identity']['class_il'] / 100.0,
            seen_average_accuracy_after=accuracy['taskwise']['class_il'] / 100.0,
            task_projection_stats=task_projection_stats,
            accuracy=accuracy,
            coverage=coverage,
            final_selected_candidate='taskwise',
            artifact_output_directory=str(output_directory),
        )
    except Exception as error:
        if classifier is not None and weight_before is not None:
            self._install_classifier_candidate(
                classifier, weight_before, bias_before,
            )
        self._record_sap_event(
            status=SAP_FAILED,
            oracle_stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        logging.exception(
            'Final task-wise Oracle SAP failed; pre-SAP classifier was restored.'
        )
```

结论：最后一个 task 的判断使用 `dataset.N_TASKS - 1`，没有硬编码 9/10；`super().end_task(dataset)` 位于 gate 前，因此 Task1～Task9 只跳过 SAP，不跳过 AER boundary。

# 4. Reference builder 修改

入口和局部变量：

```python
def _build_oracle_reference_batches(self, dataset, *, return_task_ids=False):
    ...
    buffer_images = None
    buffer_true = None
    buffer_task_ids = None
```

Old task ids 经过与 Old images/labels 完全相同的 `clean & historical` mask：

```python
buf_source_task_ids = self.buffer.task_labels[:len(buf_labels)].detach().cpu().long()
historical = buf_source_task_ids < int(self.current_task)
keep = clean & historical
buffer_images = buf_images[keep]
buffer_true = buf_true[keep]
buffer_task_ids = buf_source_task_ids[keep]
historical_buffer_clean_count = int(buffer_images.shape[0])
```

New 在既有 class-balanced sampling 完成后才生成 task ids；不会影响或重跑 sampling：

```python
reference_new_count = int(selected_task_images.shape[0])
reference_old_count = historical_buffer_clean_count
selected_task_ids = torch.full(
    (reference_new_count,), int(self.current_task), dtype=torch.long,
)
```

最终三者使用完全相同的拼接顺序：New 在前、Old 在后。

```python
if reference_old_count > 0:
    all_images = torch.cat([selected_task_images, buffer_images], dim=0)
    all_true_labels = torch.cat([selected_task_labels, buffer_true], dim=0)
    all_task_ids = torch.cat([selected_task_ids, buffer_task_ids], dim=0)
else:
    all_images = selected_task_images
    all_true_labels = selected_task_labels
    all_task_ids = selected_task_ids

...
if return_task_ids:
    return all_images, all_true_labels, all_task_ids, stats
return all_images, all_true_labels, stats
```

这里的 Old 没有额外 sampling；`buffer_task_ids` 只应用现有 Old 的 clean/historical mask。New 继续使用原来的局部 `torch.Generator`、seed、class-balanced target 和选中索引。

# 5. `X_global` / `X_t` 数据流

实际路径：

```text
_build_oracle_reference_batches(..., return_task_ids=True)  [调用 1 次]
  └─ all_images / trusted_labels / trusted_task_ids
       └─ _normalized_batches(all_images, trusted_labels)
            └─ collect_classifier_input_features(...)        [调用 1 次]
                 └─ features（final Linear 输入）
                      └─ normalize_classifier_input_features [调用 1 次]
                           └─ X_global
                                ├─ Global: X_global.T @ X_global
                                └─ Task t: X_global[trusted_task_ids == t]
```

对应代码：

```python
batches_with_labels = list(
    self._normalized_batches(all_images, trusted_labels),
)

def image_batches():
    for images, _labels in batches_with_labels:
        yield images

features = collect_classifier_input_features(
    self.net, image_batches(), total_images=total_images,
)
x_global, feature_norm_stats = normalize_classifier_input_features(features)

global_gram = x_global.transpose(0, 1) @ x_global

task_ids_device = trusted_task_ids.to(x_global.device)
for task_id in range(int(dataset.N_TASKS)):
    task_features = x_global[task_ids_device == task_id]
    task_gram = task_features.transpose(0, 1) @ task_features
```

确认：Task-wise 不重新运行 backbone，不重新抽取 feature，也不再次 L2 normalize；`X_t` 是 `X_global` 的布尔索引视图/结果。

# 6. `W_before` / Global / Task-wise 完整数据流

共同快照：

```python
classifier = resolve_classifier_module(self.net)
weight_before = classifier.weight.detach().clone()
bias_before = (
    classifier.bias.detach().clone()
    if classifier.bias is not None else None
)
```

Identity：

```python
'identity': weight_before
```

Global：

```python
global_gram = x_global.transpose(0, 1) @ x_global
(
    global_projection,
    global_energy,
    global_normalized_energy,
    global_importance,
) = self._build_oracle_projection(global_gram)
weight_after_global, global_weight_stats = project_linear_weight(
    weight_before, global_projection,
)
```

Task-wise：

```python
weight_after_taskwise = weight_before.clone()
task_projections = {}
task_ids_device = trusted_task_ids.to(x_global.device)
for task_id in range(int(dataset.N_TASKS)):
    task_features = x_global[task_ids_device == task_id]
    task_gram = task_features.transpose(0, 1) @ task_features
    task_projection, *_ = self._build_oracle_projection(task_gram)
    start_c, end_c = dataset.get_offsets(task_id)
    task_weight, _ = project_linear_weight(
        weight_before[int(start_c):int(end_c), :], task_projection,
    )
    weight_after_taskwise[int(start_c):int(end_c), :] = task_weight
    task_projections[task_id] = task_projection
```

确认：Identity 是原快照；Global 直接投影 `weight_before`；Task-wise 从 `weight_before.clone()` 开始，且每个 row block 都读取 `weight_before[start:end, :]`。三路均不读取前一路已经安装到 classifier 的权重，因此没有 Global→Task-wise 或其他串联。

# 7. Evaluation 顺序

三次 `dataset.evaluate()` 的实际调用位于 `_summarize_evaluation()`：

```python
@staticmethod
def _summarize_evaluation(dataset, model) -> dict:
    per_task_class_il, per_task_task_il = dataset.evaluate(model, dataset)
    per_task_class_il = [float(value) for value in per_task_class_il]
    per_task_task_il = [float(value) for value in per_task_task_il]
    if not per_task_class_il or not per_task_task_il:
        raise ValueError('final SAP evaluation returned no task accuracies')
    return {
        'class_il': sum(per_task_class_il) / len(per_task_class_il),
        'task_il': sum(per_task_task_il) / len(per_task_task_il),
        'per_task_class_il': per_task_class_il,
        'per_task_task_il': per_task_task_il,
    }
```

调用周围的真实代码：

```python
candidates = {
    'identity': weight_before,
    'global': weight_after_global,
    'taskwise': weight_after_taskwise,
}
accuracy = {}
for candidate_name, candidate_weight in candidates.items():
    self._install_classifier_candidate(
        classifier, candidate_weight, bias_before,
    )
    accuracy[candidate_name] = self._summarize_evaluation(dataset, self)
```

安装函数：

```python
@staticmethod
def _install_classifier_candidate(classifier, weight, bias_before) -> None:
    with torch.no_grad():
        classifier.weight.copy_(
            weight.to(device=classifier.weight.device, dtype=classifier.weight.dtype),
        )
        if bias_before is not None:
            if classifier.bias is None:
                raise ValueError('classifier bias disappeared during SAP evaluation')
            classifier.bias.copy_(
                bias_before.to(
                    device=classifier.bias.device, dtype=classifier.bias.dtype,
                ),
            )
    if bias_before is not None and not torch.equal(classifier.bias, bias_before):
        raise ValueError('classifier bias changed during SAP evaluation')
```

按 Python dict 插入顺序，三次评估前的状态分别为：

| 评估 | `classifier.weight` | `classifier.bias` |
|---|---|---|
| Identity | `weight_before` | `bias_before` |
| Global | `weight_after_global` | `bias_before` |
| Task-wise | `weight_after_taskwise` | `bias_before` |

`dataset.evaluate()` 仍是项目原实现；本 diff 未修改 evaluation、Class-IL 或 Task-IL mask。

# 8. 最终状态提交

三路 evaluate 全部完成且 artifacts 成功保存后，才运行：

```python
stage = 'taskwise_commit'
self._install_classifier_candidate(
    classifier, weight_after_taskwise, bias_before,
)
self.past_model_ckpt = copy.deepcopy(self.net.state_dict())
logging.info(
    'Task-wise SAP final selected candidate=taskwise artifacts=%s',
    output_directory,
)
```

确认：

- `W_after_taskwise` 被再次明确写回，而不是依赖第三次评估后的偶然状态。
- `_install_classifier_candidate()` 同时恢复并严格比较 `bias_before`。
- `past_model_ckpt` 仅在最终 Task-wise 写回后刷新一次；Identity、Global 和 Task-wise 构造/评估阶段均未刷新。
- 最终事件记录 `final_selected_candidate='taskwise'`。

# 9. 异常恢复路径

统一恢复代码：

```python
except Exception as error:
    if classifier is not None and weight_before is not None:
        self._install_classifier_candidate(
            classifier, weight_before, bias_before,
        )
    self._record_sap_event(
        status=SAP_FAILED,
        oracle_stage=stage,
        error_type=type(error).__name__,
        error_message=str(error),
    )
    logging.exception(
        'Final task-wise Oracle SAP failed; pre-SAP classifier was restored.'
    )
```

各指定异常的最终状态：

| 异常点 | `classifier.weight` | bias | `past_model_ckpt` |
|---|---|---|---|
| Reference 构建失败 | 尚未取得 classifier；保持进入函数时状态 | 保持原状 | 不刷新，保持 AER boundary 已有快照 |
| Reference 总数为 0 | 同上 | 保持原状 | 不刷新 |
| 某 Task empty | 已有快照，恢复 `W_before` | 恢复 `bias_before` | 不刷新 |
| Global/Task-wise projection 失败 | 恢复 `W_before`，即使内存中的候选已部分构造也不提交 | 恢复 `bias_before` | 不刷新 |
| 任一路 evaluation 失败 | 前面候选可能临时装入过 classifier，但 `except` 恢复 `W_before` | 恢复 `bias_before` | 不刷新 |
| artifact 保存失败 | 第三路评估后 classifier 暂为 Task-wise，随后由 `except` 恢复 `W_before` | 恢复 `bias_before` | 不刷新 |

函数捕获并记录 `SAP_FAILED`，不向上重新抛出。Artifact 写入不是事务性的：保存中途失败时目录中可能留下部分文件，但模型状态会按上表恢复；这不改变候选或训练状态。

# 10. Artifact 实际代码

目录构造：

```python
def _taskwise_artifact_directory(self, dataset) -> Path:
    results_root = Path(getattr(self.args, 'results_path', 'results'))
    if not results_root.is_absolute():
        results_root = Path(base_path()) / results_root
    run_id = getattr(self.args, 'conf_jobnum', None)
    if not run_id:
        run_id = f"seed{int(getattr(self.args, 'seed', 0) or 0)}"
    return (
        results_root
        / dataset.SETTING
        / dataset.NAME
        / self.NAME
        / 'final_only_taskwise_sap'
        / str(run_id)
    )
```

实际目录：

```text
<base_path>/<results_path>/<dataset.SETTING>/<dataset.NAME>/aer_sap/
  final_only_taskwise_sap/<conf_jobnum 或 seedN>/
```

保存代码：

```python
def _save_taskwise_artifacts(
    self,
    dataset,
    *,
    weight_before,
    x_global,
    trusted_labels,
    trusted_task_ids,
    coverage,
    global_projection,
    task_projections,
    weight_after_global,
    weight_after_taskwise,
    accuracy,
) -> Path:
    output_directory = self._taskwise_artifact_directory(dataset)
    output_directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        'W_before.pt': weight_before,
        'X_global.pt': x_global,
        'trusted_labels.pt': trusted_labels,
        'trusted_task_ids.pt': trusted_task_ids,
        'M_global.pt': global_projection,
        'W_after_global.pt': weight_after_global,
        'W_after_taskwise.pt': weight_after_taskwise,
    }
    for filename, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), output_directory / filename)
    for task_id, projection in sorted(task_projections.items()):
        torch.save(
            projection.detach().cpu(), output_directory / f'M_task_{task_id}.pt',
        )
    (output_directory / 'coverage.json').write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding='utf-8',
    )
    (output_directory / 'accuracy.json').write_text(
        json.dumps(accuracy, indent=2, sort_keys=True), encoding='utf-8',
    )
    return output_directory
```

文件清单：

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

确认：

- `X_global.pt` 保存变量 `x_global`，它是 `normalize_classifier_input_features(features)` 的直接返回值，即逐样本 L2-normalized feature。
- 所有 `.pt` tensor 在 `torch.save` 前均执行 `detach().cpu()`。
- `accuracy.json` 直接保存 `dataset.evaluate()` 返回的百分比及其算术平均；没有除以 100。仅兼容旧 SAP event 的 `seen_average_accuracy_*` 和 `seen_task_accuracy_comparisons` 转为 0～1。
- `coverage.json` 包含全量 `class_counts`、`task_counts`，以及每个 task 的 `expected_classes`、`present_classes`、`missing_classes`、`coverage`、`empty`，另有汇总 `empty_tasks`。

# 11. 测试逐项映射

CR-2 新增四个测试。

## `test_only_final_task_runs_taskwise_sap_after_every_aer_boundary`

断言：

- `ErAceAerAbs.end_task` 调用次数等于 `dataset.N_TASKS`，证明 T1～T9 没有跳过正常 AER boundary。
- `_run_task_boundary_sap` 仅调用一次且参数为同一 dataset，证明只有最后一个 task 进入 SAP。
- `sap_history` 恰有 `N_TASKS - 1` 个 skip event，且全部状态为 `SAP_SKIPPED_FINAL_ONLY`，证明不存在 repeated SAP。

## `test_final_taskwise_sap_uses_one_reference_and_one_feature_matrix`

断言：

- builder `assert_called_once_with(dataset, return_task_ids=True)`：Reference 只构建/采样一次。
- `collect_features.assert_called_once()` 与 `normalize_features.assert_called_once_with(x_global)`：只提取、L2 normalize 一次。
- `_build_oracle_projection.call_count == 11`：一个 Global M 加十个 Task M。
- `len(dataset.evaluated_weights) == 3`：Identity/Global/Task-wise 都完成一次 evaluation。
- 第一份评估权重等于 `W_before`；第二份等于从 `W_before` 得到的 Global；每个 Task-wise row block 等于相应 task factor 直接乘 `W_before[start:end]`：同时证明共享起点、无串联、按 row 维切分，并走 `dataset.get_offsets()`。
- 最终 `classifier.weight == taskwise_weight`，`classifier.bias == bias_before`。
- `past_model_ckpt` 每个 state-dict tensor 都等于最终 net，证明快照对应 Task-wise。
- artifact 文件集合精确匹配要求；每个 `M_task_t` shape 为 `(512, 512)`。
- `accuracy.json` 三路 overall Class-IL 分别对应 mock 的 1/2/3 百分比。
- `coverage.json` 十个 task count 均为 1 且 `empty_tasks == []`。
- 最终 SAP event 的 `final_selected_candidate == 'taskwise'`。

## `test_optional_task_ids_are_aligned_without_resampling_reference`

断言：

- `torch.randperm.call_count == 2`，恰好等于当前 task 两个 clean class 的原 sampling 次数，没有因返回 task ids 再抽样一次。
- images、labels、task_ids 长度一致。
- 前 `reference_new_count` 个 task ids 全为当前 task 1；后五个 Old task ids 全来自过滤后的 source task 0。
- Old labels 为 `[0, 0, 1, 1, 1]`，与 Old task ids 使用同一 clean/historical mask 和 New→Old 拼接边界。

## `test_reference_builder_default_return_contract_is_unchanged`

断言：

- 不传 `return_task_ids` 时结果长度仍为 3，证明 dgc-sap 旧调用路径未被强制改为四元返回。

CR-2 执行记录为：

```text
conda run -n nrgp-mammoth python -m pytest \
  tests/test_aer_sap.py tests/test_sap_oracle_smoke.py -q

18 passed in 2.59s
```

本 CR-3 只做静态 Review 和 `git diff --check`，没有重新运行训练实验。

# 12. `git diff --check`

执行：

```bash
git diff --check
```

结果：退出码 0，无输出，检查通过。

# 13. Review 结论

- 是否发现违反 CR-2 实验边界的问题：**否**。tracked diff 只修改了允许的四个文件；Reference/New:Old/sampling、L2、SAP 数学、bias、AER/buffer/training/evaluation 定义均未被改动。Global 和 Task-wise 都由同一 `W_before` 与同一 `X_global` 独立构造，最终只提交 Task-wise。
- 是否建议现在 commit：**YES**。四个针对性测试在 CR-2 已全部通过，当前 `git diff --check` 通过。
- 是否建议进入服务器完整实验：**YES，但应在 commit Review 通过后启动，并在命令中显式传入 `--sap_oracle_scale 3000`**。parser 默认值仍是冻结的现有值 100，本 diff 按要求没有改 alpha；若正式命令漏传 3000，会污染实验变量。

Review 中保留的非阻塞风险：

- 本轮按要求未跑完整 CIFAR100，真实 Task10 是否有 empty task、missing class 以及服务器端十一次 512×512 EVD 的时间/空间表现仍需正式实验验证。
- Artifact 写入不是原子操作；保存失败可能留下部分文件，但 classifier/bias/checkpoint 会按异常路径恢复。
