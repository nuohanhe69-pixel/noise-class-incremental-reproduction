# 噪声类增量学习实验交接文档（Master-ABS 对齐重跑版）

> 更新时间：2026-10-06  
> 项目：噪声类增量学习 / SAP in Noisy Class-Incremental Learning  
> 当前阶段：Master-ABS 对齐后的统一重跑已经正式启动。  
> 用途：供后续新会话、Codex CR、服务器实验与本机离线分析直接接续。  
> 原则：严格沿用当前主线与控制变量，不因单次结果好坏擅自切换研究方向。

---

## 1. 当前状态一句话

当前正式实验底座已经切换为：

```text
branch:
exp/cosface-training-loss-v1-align-master-abs-v1

commit:
89b73f4f5123c4d110f29bdd50c0fcdf93fda606
```

该 commit 的定位不是算法创新，而是：

```text
ABS replacement runtime = 原始 master ABS
+
checkpoint / resume state = 当前增强安全版本
```

基于该底座，2026-10-06 已从 fresh Task0 启动 8 路正式实验：

| 实验 | 范围 | Task1 four-settings |
|---|---|---|
| Original CE baseline | Task0 → Task9 | 不要求 |
| Normalized Cosine | Task0 → Task1 | 有 |
| Scale Cosine E1 | Task0 → Task1 | 有 |
| Scale Cosine E2 | Task0 → Task1 | 有 |
| NCE-A | Task0 → Task1 | 有 |
| NCE-B | Task0 → Task1 | 有 |
| ArcFace | Task0 → Task1 | 有 |
| CosFace | Task0 → Task9 | Task1 自动保存 |

所有旧 checkpoint 都不得作为这条 Master-ABS 新轨迹的起点。

---

## 2. 研究主线

项目目标是把 SAP（Scaled Activation Projection）从静态噪声分类迁移到 noisy Class-Incremental Learning。

当前重点不是立即重写 SAP 数学，而是诊断：

```text
样本选择
→ training / replay / ABS
→ classifier geometry
→ SAP
→ inference
→ 不同 Task 类别之间的竞争
```

长期核心问题：

> SAP 对当前 Task classifier rows 投影以后，是否改变了不同 Task 类别在全局 Class-IL 比较中的相对竞争，从而出现“当前 Task / 历史 Task 一涨一跌”的结构性现象。

所有后续分析优先使用 Experiment Diagnostic Reasoning：

```text
Measured facts
→ structural anomaly / invariant
→ execution chain
→ earliest divergence
→ 1~2 个最小假设
→ 最小反事实
→ prediction
→ 再决定是否正式修改方法
```

禁止：

```text
结果不好
→ 直接引入新 loss / 新正则 / 新 projection / 大范围 sweep
```

---

## 3. 分工

### 用户

负责：

- 服务器正式训练；
- tmux / GPU / 日志检查；
- 打包 runs / artifacts / checkpoints；
- 传回本机；
- SHA256 校验；
- 将 Codex 产出的分析文件上传给 ChatGPT；
- 不在服务器上临时改代码或扩大变量。

### Codex：代码修改

负责：

- 按 ChatGPT 给出的窄 CR 修改代码；
- targeted / regression / full tests；
- commit + push；
- 不擅自 refactor；
- 不顺手实现下一阶段；
- 一次只回答一个核心问题。

### Codex：本机离线分析

新的固定流程：

```text
服务器实验完成
→ 用户把完整数据传回本机
→ ChatGPT 指明本机输入路径和分析要求
→ Codex 只读取本机文件
→ 完成离线统计
→ 必须输出 Markdown / CSV / JSON
→ ChatGPT review 完整文件
```

默认不再让 Codex 直接 SSH 服务器进行正式离线分析。

离线分析必须至少产出：

```text
diagnostic_report.md
training_loss_master_summary.csv
baseline_task0_task9_summary.csv
task0_*.csv
task1_*.csv
analysis_manifest.json
README.md
analyze_training_loss.py
offline_analysis_bundle.tar.gz
```

不能只在聊天里返回分析正文。

### ChatGPT

负责：

- 主线维护；
- 实验设计；
- GitHub / 代码 review；
- 服务器命令；
- 控制变量核对；
- 离线分析 CR 设计；
- review Codex 的 Markdown / CSV / JSON；
- 区分 measured result / offline reconstruction / inference / hypothesis；
- 决定下一步最小反事实。

---

## 4. 服务器与环境

SSH：

```bash
ssh -p 34594 root@connect.bjb1.seetacloud.com
```

密码只使用占位符：

```text
<SERVER_PASSWORD>
```

不要把密码写入 Git、Markdown、Python、shell script、Codex CR 或分析文件。

服务器：

```text
GPU: RTX 4090
Conda env: noise-cil
Python: 3.10.21
PyTorch: 2.6.0+cu124
```

仓库：

```text
/root/autodl-tmp/hnh/noise-class-incremental-reproduction
```

数据：

```text
/root/autodl-tmp/hnh/data/
```

CIFAR-100：

```text
/root/autodl-tmp/hnh/data/CIFAR100/cifar-100-python
```

checkpoint：

```text
/root/autodl-tmp/hnh/checkpoints/
```

当前统一 batch：

```text
/root/autodl-tmp/hnh/runs/masterabs_trainingloss_20261006
```

---

## 5. 固定实验配置

| 参数 | 固定值 |
|---|---|
| dataset | seq-cifar100 |
| tasks | 10 × 10 |
| noise | symmetric 20% |
| seed | 0 |
| backbone | ResNet18 |
| base | AER + ABS / ER-ACE |
| buffer | 2000 |
| batch | 32 |
| replay batch | 32 |
| epochs | 50 / task |
| lr | 0.03 |
| optimizer | SGD |
| momentum | 0 |
| weight decay | 0 |
| Nesterov | false |
| scheduler | MultiStepLR |
| milestones | 35 / 45 |
| gamma | 0.1 |
| buffer fitting | 0 |
| use_aer | 1 |
| ABS | enabled |
| alpha_sample_insertion | 0.75 |
| SAP alpha | 300 |
| SAP batch | 32 |
| SAP reference | oracle-clean |
| SAP scope | current-task-only |
| num_workers | 4，必须显式固定 |

正式 run 始终显式使用：

```text
--base_path /root/autodl-tmp/hnh/data/
--checkpoint_path /root/autodl-tmp/hnh/checkpoints/
--num_workers 4
```

---

## 6. GitHub 仓库与关键分支

仓库：

```text
nuohanhe69-pixel/noise-class-incremental-reproduction
```

### 当前正式执行分支

```text
exp/cosface-training-loss-v1-align-master-abs-v1
89b73f4f5123c4d110f29bdd50c0fcdf93fda606
```

### 对齐前 training-loss 统一分支

```text
exp/cosface-training-loss-v1
f90335cbb15e0d96afa8273ca3c640a1c89f8ed8
```

今后该分支上的结果统一标记：

```text
pre-master-ABS historical
```

### ArcFace

```text
exp/arcface-training-loss-v1
6b802bad605559e349925f564bf640ea5262ad47
```

### NCE / double-boundary 基础

```text
feat/double-boundary-taskwise-sap-v1
472ce1316ea65e505ea0d28fa36f350e401b33f3
```

### 历史 Master-ABS 对齐参考

```text
feat/double-boundary-taskwise-sap-v1-align-master-abs-v1
6b05bcd6109976c66f64e3f2d62794f93212d14f
```

### Partial SAP

```text
exp/current-task-partial-sap-beta020
0a13cdcd9d25fa4863a7361e009831b91e0e8b19
```

### Direction-only 历史参考

```text
feat/all-seen-direction-only-sap-v1
dde649d199c7de895e5d8a9257bbbee30cc4a10c
```

注意：direction-only 历史实现是 all-seen 语义，后续不能整体 merge 到 current-task-only 主线。

---

## 7. Master-ABS alignment 的实际修改

从：

```text
f90335c...
```

到：

```text
89b73f4...
```

只修改：

```text
utils/buffer.py
tests/test_buffer_resume.py
```

以下文件没有变化：

```text
utils/checkpoints.py
models/utils/continual_model.py
models/aer_sap.py
models/er_ace_aer_abs.py
```

两处 runtime 对齐：

1. LARSSampling.normalize_scores：去掉 finite / -inf repair workaround，恢复 master 行为。
2. ABSSampling.scale_scores：historical past probability 去掉额外的 +1e-9，恢复 master 原始 denominator。

checkpoint 增强机制继续保留：

```text
num_seen_examples
sample_selection_strategy
sample_selection_state
importance_scores
buffer state
SAP state
```

目标：

```text
fresh ABS runtime == master
resume continuation == uninterrupted continuation
```

---

## 8. 为什么必须 fresh Task0

旧 checkpoint 虽然格式正确，但其中：

```text
buffer
importance_scores
num_seen_examples
```

已经属于旧 ABS trajectory。

因此不能：

```text
旧 runtime checkpoint
→ 新 Master-ABS runtime
```

当前规则：

> 所有 Master-ABS 正式结果都必须从 fresh Task0 开始产生新的 checkpoint。

---

## 9. 当前 8 路正式实验

batch：

```text
/root/autodl-tmp/hnh/runs/masterabs_trainingloss_20261006
```

### 9.1 Baseline

```text
model: er_ace_aer_abs
run: baseline_ce_task0_task9_seed0
range: Task0 -> Task9
inference: original linear
```

目的：

- 验证 Master-ABS alignment 后 baseline 最终 Class-IL 是否恢复；
- baseline 不要求 SAP four-settings。

### 9.2 Normalized Cosine

```text
run: normcos_task0_task1_seed0
range: Task0 -> Task1
four-settings: yes
```

### 9.3 Scale Cosine E1

```text
training scale=64
ABS scoring scale=1
run: scalecos_e1_task0_task1_seed0
```

### 9.4 Scale Cosine E2

```text
training scale=64
ABS scoring scale=64
run: scalecos_e2_task0_task1_seed0
```

### 9.5 NCE-A

```text
training_loss=nce
nce_ace_scope=baseline
run: nce_baseline_task0_task1_seed0
```

### 9.6 NCE-B

```text
training_loss=nce
nce_ace_scope=aligned
run: nce_aligned_task0_task1_seed0
```

### 9.7 ArcFace

```text
s=64
m=0.5
run: arcface_task0_task1_seed0
```

### 9.8 CosFace

```text
s=64
m=0.35
run: cosface_task0_task9_seed0
range: Task0 -> Task9
```

Task1 结束时仍自动产出：

```text
none
task0_only
task1_only
both
```

随后继续 Task2 → Task9。

---

## 10. 当前服务器运行状态

启动检查已经通过。

tmux：

```text
ma_arcface_t1
ma_baseline_t9
ma_cosface_t9
ma_nce_a_t1
ma_nce_b_t1
ma_normcos_t1
ma_scalecos_e1_t1
ma_scalecos_e2_t1
```

启动后确认：

```text
1 个 er-ace-aer-abs 进程
7 个 aer-sap 进程
```

RTX 4090 启动检查约：

```text
显存 ~7.4GB / 24.5GB
GPU util ~57%
```

各日志均进入首个 Task 的 Epoch 1，并确认：

```text
Noisy sym targets loaded from file
Using 4 workers for the dataloader
```

当前状态：

```text
RUNNING
```

尚未把本轮中间数字当成正式研究结论。

---

## 11. 当前实验前 predictions

### Prediction 1：baseline

如果旧 baseline 下降主要来自 ABS runtime trajectory 分叉：

```text
Master-ABS aligned fresh baseline
→ 应明显靠近最初 baseline 复现水平
```

如果仍明显偏低：

```text
ABS mismatch 不是充分解释
→ 必须按 Task 定位 earliest divergence
```

### Prediction 2：training-loss 结果

如果旧 ABS runtime 对各 training loss 影响较小：

```text
新的 qualitative pattern 应大体保持
```

如果变化很大：

> 过去部分 training-loss 结论混入了 buffer trajectory 差异。

### Prediction 3：ArcFace vs CosFace

pre-alignment 历史上：

```text
ArcFace task1_only:
old Task ↑↑
current Task ↓↓↓
overall ↓

CosFace task1_only:
old Task 小降
current Task ↑
overall ↑
```

若新 run 仍保持这一结构，则更难用 ABS 随机轨迹解释二者差异。

---

## 12. SAP reference 规则

### Task0

```text
New = 当前 Task 全部 oracle-clean
Old = 无
```

### Task1+

Old：

```text
historical buffer
source_task < current_task
observed_label == true_label
```

New：

```text
当前 Task oracle-clean
数量 = Old
类别尽量均衡
```

所以：

```text
New : Old = 1 : 1
```

但不同 loss 的 buffer trajectory 可以不同，因此实际 Old / New count 不必相同。

formal current-task-only SAP 只用当前 Task New 构造当前 Task projector。

---

## 13. Task1 four-settings 的语义

```text
none
task0_only
task1_only
both
```

本质：

```text
same Task1 boundary checkpoint
+
cosine inference
+
analysis_only counterfactual
```

four-settings 不等于四次重新训练。

CosFace Task1 分析结束后会恢复 formal network state，因此不会污染 Task2→Task9。

---

## 14. pre-Master-ABS 历史结果

以下只能作为 historical reference。

### Scale Cosine E1

```text
Task0 75.6 -> 58.7
```

### Scale Cosine E2

```text
Task0 74.3 -> 63.3
```

### NCE-A

```text
Task0 19.3 -> 18.6
```

### NCE-B

```text
Task0 57.0 -> 55.5
```

### ArcFace

```text
Task0 72.6 -> 73.0
```

Task1：

| setting | T0 Class-IL | T1 Class-IL | Overall |
|---|---:|---:|---:|
| none | 47.4 | 68.4 | 57.90 |
| task0_only | 28.4 | 73.9 | 51.15 |
| task1_only | 63.6 | 37.1 | 50.35 |
| both | 45.7 | 69.6 | 57.65 |

### CosFace

```text
Task0 77.0 -> 77.8
```

Task1：

| setting | T0 Class-IL | T1 Class-IL | Overall |
|---|---:|---:|---:|
| none | 63.1 | 66.9 | 65.00 |
| task0_only | 64.9 | 64.1 | 64.50 |
| task1_only | 62.1 | 69.6 | 65.85 |
| both | 63.8 | 67.3 | 65.55 |

新主结果必须来自本轮 89b73f4... rerun。

---

## 15. training-loss 路线的已知诊断

### Normalized Cosine

s=1 初版曾约 22.1%，核心问题包括 future classes 参与 softmax。

ACE scope 改为：

```text
current present-only
replay seen-only
```

后约恢复到 48.4%。

ABS scoring geometry 对齐后约 52.3%。

结论：

> future-class competition 是一层问题，ABS mismatch 是 downstream amplifier，但 s=1 本身仍较弱。

### Scale Cosine

s=64 明显恢复 training，但 SAP 后曾出现严重下降。

结论：

> training 动态范围问题与 SAP compatibility 是两层问题。

### NCE

NCE-A baseline scope：

```text
tiny loss
+
very poor classification
```

NCE-B aligned 明显恢复，但仍弱于 CE。

结论：

> loss 数值低不代表 representation / classifier 健康。

### ArcFace / CosFace

两者 Task0 都曾避免 Scale Cosine 那种 SAP collapse，但 Task1 interaction 明显不同。

这正是本轮 rerun 后离线分析的重点。

---

## 16. teacher cosine inference 历史结论

ordinary CE same Task1 checkpoint：

linear no-SAP overall：

```text
65.50
```

cosine no-SAP overall：

```text
66.55
```

说明 inference-only 本身可以涨点。

但 current-task SAP + cosine 曾：

```text
62.30
```

因此：

> inference 改变谁赢谁输，但没有自动解决 SAP 的跨 Task 竞争问题。

---

## 17. partial-SAP / gamma 历史诊断

partial：

```text
W(beta) = (1-beta)W + beta W_sap
```

历史 Task1 有窄 beta 区间可让 T0/T1 同时小涨。

统一 gamma scaling 也有非常窄的改善区间，但直接整体恢复 pre-SAP norm 会过冲。

结论：

> 分类器权重大小参与竞争，但简单统一缩放不是完整答案。

---

## 18. 后续 direction-only 候选

历史数学：

```text
W_full = W_before M^T

W_dir_i =
normalize(W_full_i)
*
||W_before_i||
```

只保留 SAP 后方向，每个 classifier row 单独恢复自己的 pre-SAP norm。

当前尚未在 Master-ABS 新底座正式实现。

如本轮统一分析继续支持该方向，未来 CR 必须以：

```text
exp/cosface-training-loss-v1-align-master-abs-v1
89b73f4...
```

为父节点，只迁移 row-wise norm restore，不迁移 old all-seen scope 或旧 ABS 逻辑。

---

## 19. Git 同步规则

服务器只是 execution environment。

Codex workspace 与服务器 checkout 不等价。

标准流程：

```text
git status
→ fetch target branch
→ verify remote HEAD
→ switch / ff-only
→ verify final branch + commit
```

禁止：

```text
盲目 git pull
git reset --hard
未核对 commit 就开正式实验
```

---

## 20. 当前 run 完成判定

### 六个 Task0→Task1 loss

必须确认：

```text
EXIT_STATUS=0
boundary_task_0 valid
boundary_task_1 valid
task1_four_settings.json
task1_four_settings.csv
task1_four_settings.md
```

### CosFace Task0→Task9

必须确认：

```text
EXIT_STATUS=0
Task0 ... Task9 全部完成
Task1 four-settings 存在
各 boundary artifacts 完整
```

### Baseline Task0→Task9

至少确认：

```text
EXIT_STATUS=0
Task0 ... Task9 正常评价
checkpoint / results 完整
```

baseline 不要求 SAP four-settings。

---

## 21. 服务器数据传回本机

全部 run 完成以后：

```text
completeness check
→ tar.gz
→ SHA256
→ split
→ 下载
→ 本机重组
→ SHA256 verify
→ 解压
```

必须传：

```text
整个 masterabs_trainingloss_20261006
相关 checkpoints
run logs
run_meta
SAP artifacts
Task1 four-settings
baseline Task0-Task9 metrics
CosFace Task0-Task9 artifacts
```

不要只传最终 accuracy。

---

## 22. 本机离线分析

建议根目录：

```text
~/Downloads/noise-cil-artifacts/masterabs_trainingloss_20261006
```

Codex CR 必须明确：

- 输入根目录；
- 每个 run path；
- artifact 自动发现规则；
- 必算指标；
- 对照关系；
- 输出目录；
- 必须生成的文件。

至少产出：

```text
README.md
diagnostic_report.md
analysis_manifest.json
analyze_training_loss.py

baseline_task0_task9_summary.csv
baseline_per_task_class_il.csv
baseline_per_task_task_il.csv

task0_accuracy_summary.csv
task0_geometry_summary.csv
task0_cosine_margin_summary.csv
task0_per_class_accuracy.csv
task0_per_class_margin.csv
task0_flip_summary.csv

task1_four_settings_all_losses.csv
task1_four_settings_deltas.csv
task1_cross_task_confusion.csv

sap_reference_summary.csv
buffer_summary.csv
sap_integrity_checks.csv

cosface_task0_task9_summary.csv
cosface_per_boundary_sap_delta.csv
cosface_per_task_class_il.csv
cosface_per_task_task_il.csv

training_loss_master_summary.csv
offline_analysis_bundle.tar.gz
offline_analysis_bundle.tar.gz.sha256
```

缺失数据写 NA，不得填 0。

---

## 23. 本轮离线分析核心问题

### Baseline

> Master-ABS 是否让 baseline 回到原始复现水平？

不只看 Task9 final，要找每个 Task 的 earliest divergence。

### 7 个 training loss

统一比较新 run 的：

- Task0 Pre/Post SAP；
- Task1 four-settings；
- buffer；
- reference count；
- classifier norm / direction；
- true-vs-best-wrong margin；
- prediction flips；
- cross-task confusion。

### ArcFace vs CosFace

重点回答：

> pre-alignment 的行为分叉是否在 Master-ABS 下复现？

### CosFace Task0→Task9

重点回答：

> Task1 的正收益是否能延续到完整 Class-IL 序列？

每个 boundary 看：

```text
Pre-SAP
Post-SAP
current Task delta
historical Tasks delta
overall delta
Task-IL
Class-IL
```

最终再与 Original CE baseline 对比，但必须清楚标注 training / inference 差异，不能错误 attribution。

---

## 24. 当前路线的停止条件

### training-loss 路线

本轮 rerun + 统一离线分析完成后，如果没有新的强证据支持继续增加 training loss：

```text
关闭继续开发新 loss / margin / scale 的路线
```

CosFace 即使最好，也先作为 best candidate / diagnostic reference，不自动替代 original CE 主线。

### baseline 路线

如果 Master-ABS fresh baseline 恢复：

```text
确认 ABS runtime mismatch 是重要复现因素
→ 后续再做 baseline 3-seed
```

如果没有恢复：

```text
按 Task 找 earliest divergence
```

### SAP 主线

training-loss 线收尾以后，根据证据决定是否推进：

```text
current-task-only direction-only row-norm restore
```

---

## 25. 已踩实验坑

1. Normalized Cosine s=1 + future classes 参与 softmax 会造成严重 collapse。
2. ABS scoring geometry mismatch 会放大弱类淘汰。
3. NCE loss 很低不代表分类健康。
4. Scale Cosine 修复 training 不等于修复 SAP compatibility。
5. Task-IL 稳定、Class-IL 大变时，优先怀疑跨 Task 竞争。
6. ArcFace 与 CosFace 同属 margin loss，不代表 SAP interaction 相同。
7. reference 规则相同，不代表不同 loss 的实际 Old / New count 相同。
8. 8 路并发只比较结果，不比较 wall-clock / it/s。
9. four-settings 是 analysis-only，不是四次训练。
10. CosFace Task1 four-settings 不会污染 Task2→Task9。

---

## 26. 已踩 Git / checkpoint / 服务器坑

1. 服务器 checkout 与 Codex workspace 是两个独立环境。
2. 不整体 merge 历史 direction-only / align branch。
3. git pull 曾卡住；以后 fetch → verify → ff-only。
4. 不用 reset --hard 做同步。
5. fixed seed 不保证跨代码细差异完全同轨迹。
6. ABS 的极小 probability 差异进入 np.random.choice 后可放大成整条 trajectory 分叉。
7. 旧 checkpoint 不能作为 Master-ABS 新 run 起点。
8. Task1 artifact 出现在 Task0 run path 下可能只是 resume 继承 results_path，不是异常。
9. 判断 Task1 完成不能只看 manifest 存在，要看 succeeded / artifact_complete / valid / four-settings / exit status。
10. shell 中 placeholder 文件名没替换会导致 No such file + 本地 0B 文件。
11. ssh cat 重定向无进度不代表传输卡死。
12. 大文件必须做 SHA256。

---

## 27. 当前训练期间不要做什么

当前 batch 运行期间，不要：

- 临时调 loss；
- 看 Task0 单点就停某一路；
- 改 SAP alpha；
- 开 beta sweep；
- 实现 direction-only；
- 复用旧 checkpoint；
- 开 3-seed；
- 新增新的 training-loss CR。

除非出现：

```text
OOM
process crash
artifact invalid
wrong branch / commit
wrong parameters
```

否则让当前矩阵完整跑完。

---

## 28. 后续执行顺序

1. 等待 6 个 loss 完成 Task1。
2. 等待 baseline 完成 Task9。
3. 等待 CosFace 完成 Task9。
4. 统一检查 completeness。
5. 打包所有 run + checkpoint。
6. 传回本机并 SHA256。
7. ChatGPT 产出本机 Codex offline-analysis CR。
8. Codex 只读本机数据并文件化产出。
9. ChatGPT review analysis bundle。
10. 决定：
   - baseline 是否恢复；
   - training-loss 路线是否关闭；
   - CosFace 是否保留；
   - direction-only 是否进入下一 CR；
   - baseline 3-seed 是否启动。

---

## 29. 新会话接手首先确认

新会话必须先确认：

```text
正式 branch:
exp/cosface-training-loss-v1-align-master-abs-v1

reviewed commit:
89b73f4f5123c4d110f29bdd50c0fcdf93fda606

current batch:
/root/autodl-tmp/hnh/runs/masterabs_trainingloss_20261006
```

然后检查当前 tmux / exit status，再继续，不要重复已完成实验。

---

## 30. 最终交接摘要

当前项目最重要的新变化不是增加新算法，而是：

```text
Master-ABS runtime 对齐完成
+
checkpoint resume 保留
+
所有核心 loss / baseline 从 fresh Task0 统一重跑
```

当前正式矩阵：

```text
Original CE baseline  Task0 -> Task9
CosFace               Task0 -> Task9
Normalized Cosine     Task0 -> Task1
Scale Cosine E1       Task0 -> Task1
Scale Cosine E2       Task0 -> Task1
NCE-A                 Task0 -> Task1
NCE-B                 Task0 -> Task1
ArcFace               Task0 -> Task1
```

当前优先级：

> 不再改方法，先跑完当前 Master-ABS aligned batch，完整传回本机，完成统一离线分析，再决定下一条正式路线。
