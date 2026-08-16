# SAP 实验执行清单

## 2026-08-14本机真实运行结论

- [x] CIFAR10 Symm20 Task 1真实训练：GMM固定分离度门槛导致仅10张fallback，安全跳过SAP
- [x] Task 2真实训练：两类GMM均5/5有效、30张主reference、累计fallback 25%，安全跳过SAP
- [x] Task 3真实训练：两类GMM均5/5有效、累计70张、fallback 14.29%，执行门控成功
- [x] 定位Task 3投影失败根因：MPS不支持`torch.linalg.eigh`，事务保护和checkpoint正常
- [x] 按用户要求正常SIGINT停止训练并保存暂停checkpoint
- [x] 暂停15分钟/每小时训练监控

## SAP v2下一阶段（逐阶段审核）

- [ ] Phase A1：实现仅MPS使用CPU `eigh`的设备兼容层，CUDA/CPU路径不变
- [ ] Phase A2：增加失败checkpoint的SAP-only安全重试入口，不重选reference、不覆盖原checkpoint
- [ ] Phase A3：用Task 3真实checkpoint完成10层投影、回滚/数值/非目标delta和即时准确率验证
- [ ] 用户审核Phase A后再继续
- [ ] Phase B1：实现固定1.0严格门槛 + 0.9稳定边界门槛
- [ ] Phase B2：记录每次bootstrap被拒绝的精确原因和稳定性统计
- [ ] Phase B3：用Task 1真实loss边界案例及Task 2/3高质量案例做回归测试
- [ ] 用户审核Phase B后再继续
- [ ] Phase C1：拆分pending candidate与trusted memory
- [ ] Phase C2：SAP门控与activation space只使用trusted，fallback未经晋升不参与投影
- [ ] Phase C3：checkpoint升级v2并兼容读取v1状态
- [ ] 用户审核Phase C后再继续
- [ ] Phase D1：记录epoch 35/45/50逐样本seen-class loss、预测、置信度和类内rank
- [ ] Phase D2：接入OGC状态与AER/ABS辅助证据，不使用true label做选择
- [ ] Phase D3：定义并验证pending晋升规则及合成噪声诊断纯度；GMM_MAIN首版直接trusted，只做轨迹诊断
- [ ] 用户审核Phase D后再继续
- [x] Reference预算沿用原方案：CIFAR10/CIFAR100均固定每个observed class最多15张，不设独立Phase E
- [ ] Phase F1：从头运行CIFAR10 Symm20三Task smoke并至少一次真实执行SAP
- [ ] Phase F2：审核通过后上传服务器运行CIFAR100 Symm40三Task smoke
- [ ] Phase F3：服务器smoke通过后再决定是否运行完整10 Task

## 设计审核

- [ ] 确认MVP首跑：CIFAR100 replay固定2000，SAP reference固定每类15张、总上限1500
- [ ] 确认主GMM候选不足5张时只用预测一致的最低CE样本保守补到5张，不补到15张
- [ ] 确认SAP执行门槛：当前Task覆盖9/10、至少8类有≥5张主GMM样本、累计覆盖90%、总reference≥4×已见类别、fallback≤20%
- [ ] 确认每Task内按observed class分别拟合Robust GMM与安全跳过
- [ ] 确认MVP只在每Task训练完成后评分和选择一次；epoch35/45/50轨迹筛选延期
- [ ] 确认旧Task reference成员固定，仅做诊断性复查而不自动删除
- [ ] 确认只投影layer3/layer4的10个Conv2d，论文式pre-patch，classifier关闭
- [ ] 确认主指标为各Task Class-IL准确率等权平均
- [ ] 确认SAP使用独立局部RNG，dry-run不改变后续训练随机性
- [ ] 确认reference随safe checkpoint保存和恢复

## 实现

- [ ] Task 1：SAP数学核心与矩阵单测
- [ ] 用户审核Task 1
- [ ] Task 2：ResNet hook、激活收集与权重投影
- [ ] 用户审核active layer和weight delta
- [ ] Task 3：Robust GMM与持久reference set
- [ ] 用户审核GMM质量和类别统计
- [ ] Task 4：独立`dgc-sap`、dry-run、task-boundary事务
- [ ] 用户审核parser、失败回滚和checkpoint同步

## MVP机制与首跑

- [ ] 本地短训练验证SAP执行、dry-run、回滚、checkpoint和DGC不回归
- [ ] RTX4090三Task短smoke先测试20000 patches；失败才回退10000/5000并锁定安全值
- [ ] 用户审核机制日志和最终首跑命令
- [ ] 仅运行一次CIFAR100 Symm40、seed0、DGC+SAP完整实验
- [ ] 汇总逐Task矩阵、Final ACC、SAP即时delta、GMM、投影和资源日志
- [ ] 用户审核首跑结果并决定是否补当前commit的单组DGC对照
- [ ] scale/reference/schedule网格、多seed和其他噪声条件全部延期
- [ ] 更新文档、提交、推送

## 每一步统一门槛

- [ ] 不使用test选择样本或超参数
- [ ] `git diff --check`通过
- [ ] focused tests与`tests.test_loss_trace`通过
- [ ] `dgc`训练与parser不受SAP改动影响
- [ ] 改动先审核，后提交
