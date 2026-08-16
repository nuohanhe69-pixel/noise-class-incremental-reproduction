# Baseline + DGC + SAP 具体实验方案

## 1. 实验问题

本实验回答两个分开的研究问题：

1. SAP 是否被正确接入，并确实投影了 ResNet 后两个 stage 的卷积权重？
2. 在相同 DGC 训练协议下，task-boundary SAP 是否提高最终 Class-IL 平均准确率并降低遗忘？

第一问属于机制验证，第二问属于效果验证。机制验证不通过时，不进入正式效果实验。

## 2. 固定方法与代码边界

| 方法 | 模型入口 | 训练方法 | SAP reference memory | SAP 投影 |
|---|---|---|---:|---|
| Baseline | `er-ace-aer-abs` | AER/ABS | 0 | 无 |
| DGC | `dgc` | AER/ABS + DGC | 0 | 无 |
| DGC + Ref-Control | `dgc-sap` + SAP dry-run | 与 DGC 相同 | MVP固定最终上限1500张 | 不修改权重 |
| DGC + SAP | `dgc-sap` | 与 DGC 相同 | MVP固定最终上限1500张 | `layer3`、`layer4` pre projection |

`DGC + Ref-Control` 用来检查reference筛选、保存、读取、forward/hook和局部随机数是否意外改变训练。它保存与DGC+SAP完全相同的reference，但不修改权重。由于reference不参与梯度训练、不参与replay，正确实现下它应与DGC在相同seed下逐项一致或仅有可解释的数值误差。

`DGC-Memory-Matched`不进入当前MVP。reference不参与梯度训练，它与扩大replay回答的是资源分配问题，不是SAP机制问题；只有首跑之后需要研究资源效率时再设计对应对照。

reference筛选、GMM初始化、bootstrap和activation row下采样必须使用SAP自己的局部随机数生成器，不得推进Python/NumPy/PyTorch训练全局RNG状态。dry-run前后还要校验全局RNG状态一致，避免后续Task的数据顺序或训练随机性被筛选过程改变。

AER/ABS replay与SAP reference保持两个独立集合，不能用一套选样逻辑合并：当前Task样本进入AER buffer前先经过低loss/低惩罚过滤；成为旧样本后，ABS更倾向保留其中loss较高、较易遗忘的样本，因此它近似服务于“困难但较可信样本的梯度replay”。SAP reference不参与梯度，只服务于高纯度激活子空间估计，首版仍使用低loss可信锚点。两者职责互补，但SAP的易样本子空间可能伤害AER保护的困难旧样本，因此每次SAP除评估test loaders外，还必须在投影前后对当前replay buffer逐样本计算CE，按投影前loss分为低/中/高三分位，分别报告平均loss变化、预测翻转率和准确率变化。若高loss旧buffer组显著恶化，不能把它解释成正常去噪，需优先检查trusted set代表性或投影强度。

SAP 官方参考固定为：

- 仓库：`https://github.com/sangamesh-kodge/SAP`
- master commit：`6e776ef71324d7c44afea2130b6738debdcbae8b`
- 许可证：Apache-2.0

## 3. CIFAR100数据集和固定训练协议

本实验以 Seq-CIFAR100 为基准数据集，先不同时展开 CIFAR10/Food101N。

| 项目 | 固定值 |
|---|---|
| dataset | `seq-cifar100` |
| tasks | 10，每Task 10类，共100类 |
| backbone | ResNet18 |
| noise | symmetric 40% |
| epochs | 50 / task |
| batch / minibatch | 32 / 32 |
| optimizer | SGD |
| lr | 0.03 |
| replay buffer | 2000 |
| DGC参数 | `ogc_loss_weight=0.3`、`ogc_low_conf_weight=0.4`、`ogc_buffer_penalty_coeff=1.8` |
| seeds | 开发0；正式0、1、2、3、4 |
| class order/noise cache | 同一seed的所有方法完全相同 |

方案A的SAP reference set是独立于replay buffer的额外训练侧内存。数量不能简单写成buffer的某个固定比例：buffer负责梯度replay，reference只负责估计SAP输入子空间，两者所需样本量由不同因素决定。当前MVP首跑不做预算网格，直接复用历史较优配置`reference上限1500`，即每类最多15张。500/1000预算敏感性实验全部延期，只有首跑显示SAP链路正常且结果值得继续时再考虑。报告必须明确写成 `replay 2000 + SAP reference <= 1500`，不能称为与纯DGC完全同内存预算。

历史会话 `019f1404-b55a-7fc1-839c-c13746cc8f7a` 中可直接复用的训练参数是：`batch_size=32`、`n_epochs=50`、`lr=0.03`、`buffer_size=2000`、`num_workers=4`，以及各噪声条件的DGC参数。旧实现的SAP投影位置、矩阵构造和样本筛选语义不同，因此`scale=3000`和`retain=1500`只是MVP探索性起点，不能认定为新SAP最优参数。

特别是旧代码中的`sap_retain_samples=1500`表示“每次从当前buffer的clean候选中全局最多选1500张”，会按当时已见类别重新分配；新方案表示“每类最多15张、选中后跨Task持久保存，Task 10最终最多1500张”。二者只在最终总上限数值上相同，Task 1至Task 9的reference数量和来源均不相同。

历史五组配置拆分如下；DGC三项直接作为相应噪声条件的固定起点，旧SAP scale只作为参考记录：

| 条件 | noise | 历史SAP scale | DGC loss weight | DGC low-conf weight | DGC buffer penalty |
|---|---:|---:|---:|---:|---:|
| Symm20 | 0.2 | 1000 | 0.2 | 0.5 | 1.5 |
| Symm40 | 0.4 | 3000 | 0.3 | 0.4 | 1.8 |
| Symm60 | 0.6 | 5000 | 0.3 | 0.3 | 2.0 |
| Asym20 | 0.2 | 1000 | 0.2 | 0.5 | 1.5 |
| Asym40 | 0.4 | 3000 | 0.3 | 0.4 | 1.8 |

当前主条件暂定为symmetric 40%。锁定SAP实现和scale后，再按算力依次扩展symmetric 20%与60%；每个噪声比例使用`readme_latest.md`中对应的DGC参数，不能在不同方法间单独调DGC。若本项目最终主表只要求一个噪声条件，必须在正式运行前明确锁定，不能看结果后更换主条件。

## 4. SAP reference set如何建立

### 4.1 建立时间

每个Task完成全部epoch和原有DGC/AER `end_task` 后，先在当前Task完整训练集上建立该Task的reference，再用累计reference执行SAP。

旧Task reference不会在后续Task中重新按当前loss筛掉；这样避免遗忘造成早期Task loss整体升高而被淘汰，也避免依赖非均衡replay buffer。

### 4.2 当前Task评分

当前Task完成全部50个epoch及原有DGC/AER Task结束处理后，只对当前Task完整训练集执行一次确定性的评分forward：

```text
model.eval() + torch.no_grad()
shuffle=False
not_aug_input + normalization
不使用随机裁剪/翻转
CE(logits over all seen classes, observed noisy label)
```

其中Task `t`只保留从第0类到当前Task结束类别的logits，尚未学习的future classes不参与softmax。Task 1时seen-class CE等同于task-local CE；Task 2以后，它还能利用旧类logits暴露“真实图像像旧类、observed label却落在当前Task”的跨Task噪声。该分数有意不同于DGC当前样本训练时的ER-ACE特殊mask：训练mask用于降低旧类偏置，SAP分数用于检查observed label在已学类别中的可信度。另行记录task-local CE、observed-label confidence、prediction和true-label诊断；真正筛选只使用训练完成后的最终seen-class CE及其GMM概率，`true_labels`绝不参与筛选。
其中Task `t`只保留从第0类到当前Task结束类别的logits，尚未学习的future classes不参与softmax。Task 1时seen-class CE等同于task-local CE；Task 2以后，它还能利用旧类logits暴露“真实图像像旧类、observed label却落在当前Task”的跨Task噪声。该分数有意不同于DGC当前样本训练时的ER-ACE特殊mask：训练mask用于降低旧类偏置，SAP分数用于检查observed label在已学类别中的可信度。另行记录task-local CE、observed-label confidence、prediction和true-label诊断；真正筛选只使用最终seen-class CE、预测一致性及其GMM概率，`true_labels`绝不参与筛选。

### 4.3 Robust GMM

每个Task使用训练完成后的最终seen-class CE，再按observed class拆分；本Task的10个类别分别拟合双分量GMM，而不是把10类loss混在一起。MVP完整首跑不划validation，CIFAR100原始每类500张，注入噪声后每个observed class仍约有500张（转入/转出会有波动），足够进行类别内双分量拟合。按类拟合可避免本来就更难的类别因整体loss较高而在跨类阈值下被误删。

当前项目的数据顺序是：若启用validation则先按clean label划分，再对剩余训练标签注入噪声，最后按noisy observed label划分Task。因此“Task 3训练集”指observed label落在Task 3类别范围内的样本，个别样本的true class可能来自其他Task。SAP分组和类别配额以observed label为准，筛选分数为seen-class CE；true label只用于合成噪声下的离线纯度诊断，不得修正Task归属或参与选择。

每个类别的GMM流程：

1. 对loss做1%/99%截尾并缩放到 `[0,1]`；
2. `n_components=2`、`n_init=10`、`reg_covar=1e-4`、固定seed；
3. 低均值分量定义为clean分量；
4. 固定该类完整loss的截尾和缩放边界；使用SAP局部RNG进行5次有放回bootstrap，每次重采样该类同样数量的loss拟合GMM，再对该类全部原始样本计算clean posterior；
5. 可信候选需满足：平均 `P(clean)>=0.9`、至少4/5次属于clean、概率标准差不超过0.1；
6. 检查收敛、loss标准差、分量权重及标准化均值分离度。

GMM质量初始门槛：

```text
min_samples=100
min_loss_std=0.02
min_component_weight=0.05
min_separation=1.0
fallback=skip
```

主选择仍使用上述GMM可信条件。为避免某一类GMM退化后永久没有reference，增加一个与SAP原论文最低loss原则一致的保守fallback：若该类有效bootstrap少于4次，或主候选少于5张，先保留已有主候选，再只从剩余的“当前模型seen-class预测等于observed label”样本中按原始seen-class CE升序补到总计最多5张。fallback不补到15张，不采用随机样本，不使用true label；若预测一致样本仍不足5张，只保留实际数量并记录`GMM_FALLBACK_LOW_LOSS`。其他类别不能占用该类配额。

### 4.4 每类数量与持久保存

CIFAR100 reference使用按类等额配额。当前MVP固定每类最多15张，总上限1500：

```text
Task 1结束：10类，各最多q张，累计最多10q
Task 2结束：20类，各最多q张，累计最多20q
...
Task 10结束：100类，各最多q张，累计最多100q
q = 15，最终上限1500
```

在可信候选内按原始seen-class CE升序，每类取最多q张。某类不足q张时只保存实际可信数量，不用高loss样本补齐，也不用其他类抢占该类配额。

“旧Task reference永久保留”只表示在同一次持续学习运行中，一旦某个Task在其刚训练完成、模型对它最熟悉时选出reference，后续Task不再用后来的模型loss重新淘汰它；它不参与梯度训练，也不是跨不同seed/实验共享。后续Task可以用当前模型对旧reference做诊断性复查，记录其当前seen-class loss、预测是否仍与observed label一致，以及是否更像新学到的类别，但MVP不据此删除或替换，避免把模型遗忘误当成标签噪声。以Task 3结束为例：

1. Task 1结束时只从Task 1完整训练部分筛出`R1`并保存；
2. Task 2结束时保留`R1`不重评分，只从Task 2筛出`R2`，用`R1 ∪ R2`执行SAP；
3. Task 3结束时保留`R1`、`R2`不重评分，只让Task 3完整训练部分通过当前模型计算seen-class CE；按observed class拆成10组、分别拟合GMM，再从每类可信候选中各取最多q张形成`R3`；
4. 随后用冻结的SAP前source model让累计集合`R1 ∪ R2 ∪ R3`前向，对layer3/layer4逐层建立输入patch统计并投影candidate model；
5. 若`q=15`且各类都足额，Task 3共使用450张reference，其中前300张来自Task 1/2，新增150张来自Task 3。

这样做的目的不是假设旧reference永远低loss，而是防止模型遗忘导致Task 1样本在Task 3时loss整体升高，从而被一个跨Task统一阈值错误删除。

每条reference保存：

```text
not_aug_input（CPU uint8原图；进入模型前再ToTensor和Normalize）
observed_label
source_task_id
sample_id
selection_loss
clean_probability_mean/std/votes
final seen-class loss / prediction / confidence
selection_source（GMM_MAIN或GMM_FALLBACK_LOW_LOSS）
```

训练轨迹方案（例如epoch 35/45/50多点评分）可以缓解模型在末期记忆噪声的问题，但它改变了原始SAP“训练完成后选择trusted data”的流程。当前MVP不实现；只有最终单次SAP正常执行但reference纯度诊断显示未来类噪声误选严重时，再作为第二版增强单独设计和比较。

reference set及其元数据必须进入safe checkpoint。恢复训练时直接恢复已经选定的旧Task reference，不能用恢复后的模型重新筛选旧Task，否则中断前后实验不等价。

CIFAR100单张原图为`3×32×32`。最终1500张若按CPU uint8保存约占4.6MB（不含少量元数据）；reference不常驻GPU，只按batch送入。因此reference预算主要由统计覆盖和实验公平性决定，不由24GB显存直接决定；GPU峰值主要来自patch统计和矩阵分解。

CIFAR合成噪声的true label只保存到诊断日志，不作为reference运行所必需的字段。

### 4.5 SAP最低执行门槛

MVP初始规则：

- 当前新Task的10个observed classes至少覆盖9个；
- 当前新Task至少8/10个类别各有不少于5张主GMM可信样本，不能只靠fallback达到覆盖门槛；
- 累计已见类别reference覆盖率至少90%，其中“覆盖”表示该类至少1张；
- reference总数至少 `4 × 已见类别数`；
- 累计reference中fallback来源占比不超过20%；超过时记录`SAP_SKIPPED_EXCESSIVE_FALLBACK`；
- 不再因单个历史类别为0而让后续所有SAP永久跳过；缺失类别必须单独记录并检查其SAP前后准确率；
- 不满足时记录 `SAP_SKIPPED_INSUFFICIENT_REFERENCE`，模型权重不变，但已选reference仍保存到checkpoint。

## 5. Reference样本如何进入模型并执行SAP

在当前Task reference加入累计集合后：

1. 冻结SAP前的source model，并复制一个只接收投影的candidate model；
2. 两个模型都设为eval；
3. 每次只给一个目标Conv2d注册`forward_pre_hook`，累计reference按batch=32通过冻结source model；
4. 根据该卷积的kernel、stride、padding、dilation对输入执行`unfold`，得到维度为 `C_in×K_h×K_w` 的patch vectors；每层按图片均衡、固定seed最多采样20,000个patch；
5. 移除当前层hook；
6. 每个卷积层独立求输入空间奇异向量。正式实现优先按batch流式累计 `G=X^T X`，再对`G`做特征分解；其特征向量等于patch矩阵`X`的右奇异向量，奇异值为特征值平方根。小矩阵单测必须与直接SVD构造的投影矩阵一致；
7. 按官方SAP奇异值能量缩放公式构造 `Mr`；
8. 按论文公式对candidate对应卷积执行 `W_new = W_flat @ Mr^T`；
9. 不投影`conv1/layer1/layer2/classifier`；
10. 释放当前层patch/SVD矩阵，再处理下一层；所有层的表示都来自同一个未修改source model；
11. 验证通过后原位加载candidate state_dict，并同步AER `past_model_ckpt`。

ResNet18预计目标卷积为10个：

```text
layer3.0.conv1 / conv2 / shortcut.0
layer3.1.conv1 / conv2
layer4.0.conv1 / conv2 / shortcut.0
layer4.1.conv1 / conv2
```

MVP首跑SAP设置：

```text
projection_location=pre
projection_type=Mr
scale_coff=3000（复用历史buffer=2000的Symm40较优配置）
project_classifier=0
max_activation_patches=20000
```

在Task 10最终1500张reference下，layer3每层最多产生约96,000个patch，按20,000上限采样；layer4每层约24,000个patch，也按20,000上限采样。最大3×3卷积的patch维度为4608，若显式保存float32的 `20,000 × 4,608` 矩阵，本身约为369MB（约352MiB），直接SVD还会产生额外工作区。流式累计`X^T X`可把长期保存的统计矩阵降为`4,608 × 4,608`，约85MB，但特征分解耗时仍需实测。直接SVD与流式Gram的投影矩阵等价性先在单测和中等维度层验证；最大4608维层默认只基准流式Gram，只有资源允许时才额外跑直接SVD。若20,000不可承受，应在MVP机制测试中统一锁定更小上限，不能在完整首跑途中临时改变。

## 5.1 RTX 4090实验服务器运行约束

机器基线统一采用历史会话`019f1404-b55a-7fc1-839c-c13746cc8f7a`：单张RTX 4090，GPU 0，总显存约`24564 MiB`；WSL项目目录为`/home/hnh/mammoth_cifar100/mammoth_code`，Conda环境为`/home/hnh/conda_envs/nrgp-mammoth`，初始化脚本为`/home/hnh/software/miniconda3/etc/profile.d/conda.sh`。所有数据、cache、checkpoint、日志和临时文件仅写入`/home/hnh`范围。每次正式批次开始前仍把GPU名称、显存、驱动、PyTorch/CUDA版本写入实验manifest，防止服务器后续换卡或环境变化却沿用旧记录。

历史记录已经验证普通CIFAR100训练可在同一张4090上运行三进程，截图当时总显存约`3342 MiB`；但这个结果来自旧SAP实现，不能直接证明新pre-patch统计和矩阵分解也能三并行。执行策略固定为：

1. MVP机制验证和唯一完整首跑一律单进程，不安排并行；
2. 三Task短smoke先使用`max_activation_patches=20000`并记录普通训练峰值、SAP峰值、最慢层耗时和Task边界总耗时；
3. 若20000资源失败，才依次回退10000、5000；第一个通过机制与资源门槛的最大值在完整首跑前锁死；
4. 不能在完整首跑途中因OOM临时改变batch、patch上限或reference预算；
5. `batch_size=32`、`num_workers=4`沿用历史稳定值，短smoke可用`num_workers=0`排错，但完整首跑恢复为4。

任何active layer为0、投影矩阵为0、目标权重变化为0、出现NaN/Inf或非目标参数变化时，都视为SAP失败，原模型保持不变。

CIFAR100共有10个task，连续执行10次投影可能产生累积收缩。MVP首跑固定`every_task`，不增加调度消融；每次必须额外记录每层投影矩阵的最小/最大特征值、trace、有效秩、权重范数比和相对weight delta，若数值检查失败则事务式回滚。`final_task_only`延期到首跑显示有必要继续研究时。

## 6. 评估口径

每个Task结束并执行SAP后，使用项目原有的独立test loaders评估所有已见Task。主指标是Class-IL：预测时在全部已见类别中竞争。

Task 10后的最终主指标：

```text
Final ACC = mean(Acc_T1, Acc_T2, ..., Acc_T10)
```

这是Task等权macro average，不是合并打乱全部测试样本的micro accuracy。

同时报告：

- 每个Task边界后的完整Class-IL准确率矩阵；
- 最终各Task准确率；
- Final Class-IL mean accuracy；
- Task-IL mean accuracy（辅助）；
- forgetting；
- backward transfer；
- SAP前后即时准确率变化 `ΔAcc_t`；
- 每个Task SAP执行/跳过状态。

测试集不能用于筛选reference、scale或projection参数。

## 7. 当前MVP：实现后只跑一组

当前目标不是完成论文级消融，而是先回答“新的论文式SAP能否正确运行，以及单次结果是否值得继续”。执行分两步。

### 7.1 本地机制验证（不算正式实验组）

必须先通过数学单测、hook/shape单测和一段短训练smoke。最低验收：

- DGC不加载任何SAP代码路径时行为不变；
- dry-run不修改权重、不推进训练全局RNG；
- 实际匹配layer3/layer4的10个卷积；
- 只有目标卷积权重变化，非目标参数delta为0；
- logits变化非零且有限；
- reference、GMM、checkpoint恢复和失败回滚可工作；
- 使用三Task短smoke验证R1/R2/R3持久保存、三次Task边界SAP和RTX4090资源；优先20000 patches，失败时才回退10000、5000。

每完成一个实现步骤即停止，等待用户审核，不连续完成全部代码。

### 7.2 唯一一次CIFAR100完整首跑

完整首跑固定为：

```text
dataset=seq-cifar100
noise_type=symm
noise_rate=0.4
model=dgc-sap
backbone=resnet18
seed=0
n_epochs=50
batch_size=32
minibatch_size=32
lr=0.03
buffer_size=2000
num_workers=4
ogc_loss_weight=0.3
ogc_low_conf_weight=0.4
ogc_buffer_penalty_coeff=1.8
sap_scale_coff=3000
sap_reference_per_class=15
sap_reference_max=1500
sap_projection=Mr
sap_location=pre
sap_layers=layer3,layer4
sap_every_task=1
sap_project_classifier=0
sap_max_activation_patches=20000（若机制测试不通过则改为测试确认的最大安全值）
```

参数来源：历史`buffer=2000`的Symm40可比记录中，`scale=3000 / retain=1500 / DGC=0.3,0.4,1.8 / seed=0`得到过Class-IL `40.29%`、Task-IL `69.54%`。历史绝对最高`42.85%`使用`buffer=2500`和旧CBP，不用于本次首跑，因为它改变了replay预算且不是新SAP。

本次不额外运行Baseline、纯DGC、多seed、scale网格、reference网格或调度消融。历史结果只作为粗参考，不能据此声明新SAP优于当前DGC；因为代码已经删除旧CBP并重构DGC，正式因果比较仍需要未来补跑同commit的DGC对照。

单次首跑仍必须输出：

- 每个Task的Class-IL/Task-IL和最终10-Task平均；
- 每次SAP前后所有已见Task的即时准确率差；
- 每类GMM状态、可信候选数和实际reference数；
- 每层patch数、输入维度、分解秩、投影特征值和weight delta；
- 每次SAP执行/跳过/回滚原因；
- 峰值显存和SAP耗时；
- Git commit、noise cache、seed和完整命令。

首跑完成后只做一次决策：

1. 若SAP多次未执行、权重delta为0、出现NaN/Inf或频繁回滚，先修实现，不追加实验；
2. 若SAP正常执行但最终结果明显差，同时多数Task在SAP后立即下降，先分析投影强度和重复投影，不盲目扩网格；
3. 若SAP正常且结果接近或高于历史DGC区间，下一次只补当前commit下同seed、同noise cache的纯DGC一组，形成最小配对比较；
4. 多seed和其他噪声条件全部延期到这两组结果审核之后。

## 8. 记录和验收标准

每个Task至少记录：

```text
task_id
GMM converged / means / std / weights / separation
trusted candidates per class
selected reference per class
reference purity（仅synthetic诊断）
active layer keys
activation matrix shape / sampled rows / SVD rank
projection finite status
projection eigenvalue range / trace / effective rank
per-layer weight delta norm
per-layer weight norm ratio
non-target weight delta norm
logits delta norm
SAP status/reason
SAP前后每个已见Task的Class-IL accuracy
```

机制验收：10个目标卷积全部可见、目标weight delta>0、非目标delta=0、无NaN/Inf、dry-run等同DGC。

效果验收不预设“必须提升”；如SAP无提升，仍保留完整负结果和诊断，不能事后改变评估口径或删除失败seed。

## 9. 实现与人工审核顺序

1. 官方SAP数学核心与矩阵单测；审核后提交。
2. ResNet layer3/layer4 hook和投影单测；审核后提交。
3. Robust GMM和reference set管理；审核后提交。
4. 独立`dgc-sap`模型、dry-run和事务式task-boundary；审核后提交。
5. 本地短训练、RTX4090 patch/显存机制验证；用户审核日志。
6. 锁定上述唯一命令，运行一次CIFAR100 Symm40 DGC+SAP。
7. 汇总首跑结果，由用户决定是否补一组当前commit的DGC对照。

每一步完成后停止，等待用户审核，不跨步实施。

## 10. CIFAR10 Symm20真实运行后的SAP v2改进计划（2026-08-14）

### 10.1 已获得的事实与停止点

本机真实CIFAR10 Symm20、seed 0运行到Task 4训练中途后人工停止。有效证据如下：

- Task 1：两类GMM均正常形成约80/20双分量，但5次标准化均值分离度为约0.94～0.98，略低于固定`min_separation=1.0`，因此0个主样本；每类保存5个预测一致最低loss fallback，诊断纯度1.0；SAP因fallback=100%跳过。
- Task 2：两类bootstrap GMM均5/5有效，每类选15个主样本，诊断纯度1.0；累计40个reference中10个fallback占25%，SAP继续跳过。
- Task 3：两类bootstrap GMM均5/5有效，每类选15个主样本，累计70个reference、fallback占14.29%，执行门控成功；真正投影在首个`torch.linalg.eigh`处因MPS不实现该算子而事务式失败，原模型权重保留。
- Task 3正式checkpoint与SIGINT暂停checkpoint均可读取，模型参数有限，SAP历史为两次安全跳过和一次`SAP_FAILED`，reference按Task为10/30/30。

这组运行已经证明：最终单次GMM不必然在每个Task失效，累计门控可以在Task 3放行；当前首要阻塞是投影设备兼容。选样改进仍然必要，但不能与设备修复混在同一个验证阶段。

### 10.2 Phase A：投影设备兼容与真实checkpoint重试（最高优先级）

**目标：** 不改变选样、配额、门控和投影公式，只修复MPS上`torch.linalg.eigh`不可用的问题，并用已经保存的Task 3模型与70张真实reference验证完整10层投影。

实现要求：

1. `symmetric_gram`在CUDA/CPU上保持原设备计算；仅当设备为MPS时，将特征分解输入显式搬到CPU，完成`eigh`后把特征值、特征向量和投影矩阵搬回原设备与原dtype。
2. 记录每层Gram设备、分解设备、dtype、维度、耗时和峰值内存；不得通过全局环境变量静默改变其他算子。
3. 新增“从失败checkpoint仅重试SAP事务”的只读输入/新输出入口：直接使用checkpoint内已有reference，不重新筛选、不重复`add_task`、不覆盖原checkpoint。
4. 重试成功后生成派生checkpoint；同步`past_model_ckpt`，把原`SAP_FAILED`保留为历史证据，并追加明确的retry成功事件。
5. 继续保留candidate事务、异常回滚、非目标参数delta=0和数值有限性检查。

验收标准：

- 小矩阵CPU结果与现有直接实现数值一致；有MPS时通过MPS→CPU→MPS设备测试。
- Task 3真实checkpoint完成10个目标卷积，所有目标weight delta>0、非目标state delta=0、无NaN/Inf。
- 输出Task 1～3投影前后即时准确率、70张reference和500张replay的logits/loss三分位变化。
- 用户审核Phase A结果后才进入选样v2。

### 10.3 Phase B：GMM有效性从固定门槛改为稳定性门槛

**目标：** 处理Task 1这种“所有bootstrap均收敛且分量比例合理，但分离度稳定地略低于1.0”的边界情况；不能为了让SAP执行而无条件降低阈值。

拟定规则：

- 严格通过：bootstrap分离度中位数`>=1.0`。
- 稳定边界通过：中位数`>=0.9`，至少4/5次收敛，所有有效分量权重`>=0.05`，分离度标准差`<=0.03`，低均值分量身份一致。
- 硬拒绝：中位数`<0.9`、分量塌缩、拟合不稳定或非有限。
- 样本级条件仍保持平均`P(clean)>=0.9`、clean vote至少4/5、posterior标准差`<=0.1`、最终预测与observed label一致。
- 永久日志必须区分预检查失败、收敛失败、权重失败、分离度失败和样本posterior失败，不能再只记录`valid_fits=0`。

验收标准：

- 使用本次Task 1实际loss统计构造回归测试：稳定0.94～0.98可以进入样本级筛选；不稳定或低于0.9仍拒绝。
- Task 2/3原有高质量GMM结果不退化。
- true label仅用于离线纯度诊断，不参与门槛或样本选择。

### 10.4 Phase C：candidate、trusted与SAP执行三级分离

**目标：** 修复当前fallback虽然没有通过GMM、却直接进入永久reference memory并可能在未来参与投影的语义混淆。

状态设计：

```text
GMM_MAIN / 多证据通过样本 -> trusted memory -> 可参与activation space
最低loss预测一致fallback -> pending candidate memory -> 暂不参与投影
门控 -> 仅根据trusted memory判断并执行SAP
```

具体要求：

- pending仍保存CPU uint8图像、observed label、Task/sample id和完整选择证据，以便后续晋升或审计。
- 旧trusted成员在同一次运行中保持不可变，避免遗忘导致其被重新淘汰。
- pending不能计入trusted覆盖、数量和fallback比例；若晋升，记录晋升依据与时间。
- checkpoint状态升级为v2，并支持读取当前v1 checkpoint进行诊断；不得静默丢弃历史reference。
- 如果当前Task trusted不足，SAP安全跳过，但pending与诊断状态仍保存。

### 10.5 Phase D：结合AER/OGC的多阶段可信证据

**主要目标：** 为Phase C中的pending fallback提供独立于最终单点loss的晋升证据。GMM_MAIN仍直接进入trusted memory；首版对GMM_MAIN只记录多阶段诊断，不增加额外准入硬门槛。这样可以先验证fallback晋升机制，而不让高质量GMM主样本因新规则被意外拒绝。

同时避免只在最后一个epoch用单点loss判断，降低模型末期记忆噪声或困难干净样本被误判的风险。AER在当前代码中主要切换旧buffer replay，当前Task每个epoch仍训练，因此不能把“AER后高loss”直接解释为噪声。

MVP评分时间采用1-based epoch 35、45、50。每次对当前Task完整原图做确定性seen-class评分，按sample id累计：

- seen-class CE与类内loss rank；
- observed-label预测一致性与置信度；
- OGC高/低置信状态及动态阈值；
- AER/ABS当前样本低分插入资格或等价综合分数；
- 三次均值、标准差和趋势。

pending晋升为多证据trusted的初始条件需在实现前单独审核，候选方向为：三次预测一致、至少2/3次处于类内低loss区域、至少2/3次通过OGC置信门槛、轨迹波动受控。AER/ABS只能作为辅助证据，不能单独决定clean；旧buffer高loss也不能直接判为noisy，因为ABS有意保留困难旧样本。晋升后仍受该observed class的15张总配额限制。

先实现轨迹记录和诊断，再根据合成噪声纯度离线评估阈值；不能看test accuracy反向选择样本规则。

### 10.6 Reference预算固定：沿用每类15张，不设独立Phase E

用户已确认此前把5个Task误解为5个类别；当前不再增加预算选择或敏感性实验，CIFAR10和CIFAR100均沿用`每个observed class最多15张trusted reference`。该预算与replay buffer完全独立。

固定比例为：

- CIFAR10：50,000张训练图、10类、每类约5,000张；15/类约为该类0.3%。每Task 2类，理想新增30张；5个Task最终最多150张，也是全训练集0.3%。
- CIFAR100：50,000张训练图、100类、每类500张；15/类为该类3%。每Task 10类，理想新增150张；10个Task最终最多1,500张，也是全训练集3%。
- CIFAR10 replay buffer仍为500，CIFAR100 replay buffer仍为2,000；它们参与梯度训练，SAP reference不参与梯度训练，不能用二者相除来定义reference比例。

每类配额独立，其他类别不能占用空余名额。GMM_MAIN和由pending晋升的多证据trusted合计最多15张/类；不足15时保存实际可信数量，不用随机样本或高loss样本补齐。门控仅统计trusted；pending不计入trusted数量、覆盖或activation space。

### 10.7 Phase F：重新验证与服务器放行顺序

1. Phase A focused tests + Task 3真实checkpoint重试；用户审核。
2. Phase B真实Task 1 loss回归 + Task 2/3不回归；用户审核。
3. Phase C checkpoint迁移、pending/trusted晋升与门控测试；用户审核。
4. Phase D真实CIFAR10短训练，检查35/45/50轨迹与合成噪声诊断纯度；用户审核。
5. 从头运行CIFAR10 Symm20三Task smoke，必须至少一次真实`SAP_EXECUTED`、10层投影完整、非目标delta=0；用户审核。
6. 才上传RTX服务器运行CIFAR100 Symm40三Task smoke；通过后决定是否完整10 Task。

任何阶段不得用test集选择GMM、轨迹阈值、reference预算或SAP scale；test只报告投影前后效果。
