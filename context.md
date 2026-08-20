# 项目交接总结

> 本文件是跨会话共享的**项目现状快照**，记录当前目标、最新实验结论和下一步计划。
> 更新方式是**整体替换**，不追加历史；已经稳定的结论沉淀到 `AGENTS.md` 或 `readme_latest.md` 后应从这里删除。
> 稳定的操作规则、约束和环境说明见 `AGENTS.md`，此处不重复。

## 当前阶段

SAP oracle 上限分支代码已完成并通过验证（尚未提交）。CIFAR-10 symm20 正式实验已于 2026-08-20 10:41 启动，由 LaunchAgent `com.hunk.cifar10saporacle.20260820` 托管，每 30 分钟监控一次。

- 分支：`SAP`（基于 commit `e4ef125`）
- 核心变更：`dgc-sap` 新增 `--sap_oracle_reference 1` 上限模式
- 验证状态：61 个单元测试全部通过（含 13 个新增 oracle 测试）
- 工作区状态：`AGENTS.md`、`models/dgc_sap.py`、`readme_latest.md`、`utils/sap.py` 有未提交修改，另有 `tests/test_sap_oracle_classifier.py`、`tests/test_sap_oracle_smoke.py` 两个未跟踪测试文件

## oracle 模式定义（本轮上限实验）

每个 task 的第 50 个 epoch 结束后，用以下样本做 SAP：

- 当前 task 全部训练样本中 `observed == true_labels` 的干净样本
- buffer 中旧 task（`task_id != current_task`）且 `observed == true_labels` 的干净样本

关键约束：

- 不走安全门，不做 Robust GMM / promotion / reference 多条件筛选
- 保留全部特征方向（512 维完整特征分解，不截断）
- 只投影最终 classifier 的 `nn.Linear` 权重（`W' = W @ Mᵀ`），bias 不动
- 事件状态记为 `SAP_ORACLE_EXECUTED`
- 参数：`--sap_oracle_scale 100`（默认值；512 维 Gram 下 30000 会近似恒等）

## 实验结果一览

| 实验 | 设置要点 | 结果 | 结论 |
|---|---|---|---|
| DGC+SAP v3 | CIFAR-10 symm20，5 task，conv 投影（layer3/layer4 十层），带安全门 | Class-IL 65.15%，Task-IL 91.07% | 当前正式基线；SAP 各 task 均成功执行，无 NaN/回滚 |
| DGC+SAP oracle | CIFAR-10 symm20，oracle 参考（当前 task 干净样本 + 旧 task 干净 buffer），仅投影 Linear，scale=100，无安全门 | 运行中（PID 26494，2026-08-20 10:41 启动） | 日志：`run_logs/cifar10_symm20_dgc_sap_oracle_20260820/train.log`；checkpoint：`checkpoints/cifar10_symm20_dgc_sap_oracle_20260820` |

## 待定问题

1. `models/dgc_sap.py` 中 `build_sap_projection_from_gram` 被 import 但未使用，oracle 投影是自己重新实现的。建议删除 import 或改为复用，避免两处逻辑漂移。
2. oracle 实验同时改了两个变量（参考集 oracle 化 + 投影层 conv→Linear），如结果有信号需要补"conv + oracle 参考"对照臂分离归因。
3. 已有监控/核验脚本按 `SAP_EXECUTED` 找记录，跑 oracle 实验前需确认也识别 `SAP_ORACLE_EXECUTED`，否则会误报 SAP 未执行。

## 下一步

1. 监控 oracle 正式实验至 5 个 task 完成，重点核对每个 task 的 `SAP_ORACLE_EXECUTED` 事件、参考集规模、权重范数比和 checkpoint 可读性。
2. 实验完成后结果对照 v3 基线 Class-IL 65.15%：
   - 基本持平 → 选样不是瓶颈，瓶颈在机制或超参
   - 明显更高 → trusted data 选择有改进空间
   - 更低 → Linear-only 投影弱于 conv 投影，需重新审视层选择
