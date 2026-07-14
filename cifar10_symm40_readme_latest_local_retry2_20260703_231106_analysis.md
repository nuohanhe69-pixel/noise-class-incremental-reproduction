# CIFAR10 Symm40 本机重跑结果分析

## 1. 实验状态

- RUN_ID: `cifar10_symm40_readme_latest_local_retry2_20260703_231106`
- 项目目录: `/Users/hunk/Documents/deep learning/噪声类增量项目复现/mammoth_code`
- 运行状态: 已完成，PID `30004` 已退出
- 最终日志时间: `04-Jul-26 02:25:46`
- 日志文件: `run_logs/cifar10_symm40_readme_latest_local_retry2_20260703_231106.log`
- 结果目录: `data/results_cifar10_symm40_readme_latest_local_retry2_20260703_231106`
- checkpoint: `checkpoints/cifar10_symm40_readme_latest_local_retry2_20260703_231106/cifar10_symm40_readme_latest_local_retry2_20260703_231106_ogc-sap_seq-cifar10_None_500_50_20260703-231108_10304bea_last.pt`
- 异常检查: 未发现 `Traceback`、`RuntimeError`、`Exception` 或 fatal error。

## 2. 本次运行参数

本次使用 `readme_latest.md` 中 CIFAR10 Symmetric 40% 推荐参数，并在本机 MPS 上运行。

```bash
/Users/hunk/miniforge3/envs/nrgp-mammoth/bin/python -u main.py \
  --dataset seq-cifar10 \
  --model ogc-sap \
  --noise_rate 0.4 \
  --noise_type symm \
  --sap_scale_coff 2000 \
  --ogc_loss_weight 0.40 \
  --ogc_low_conf_weight 0.30 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 400 \
  --backbone resnet18 \
  --n_epochs 50 \
  --batch_size 32 \
  --lr 0.03 \
  --buffer_size 500 \
  --num_workers 4 \
  --savecheck last \
  --seed 0 \
  --results_path results_cifar10_symm40_readme_latest_local_retry2_20260703_231106 \
  --checkpoint_path checkpoints/cifar10_symm40_readme_latest_local_retry2_20260703_231106 \
  --ckpt_name cifar10_symm40_readme_latest_local_retry2_20260703_231106
```

关键默认参数记录如下：

| 类别 | 参数 |
| --- | --- |
| 数据集 | `seq-cifar10`, 5 tasks, 10 classes, symmetric label noise |
| 噪声 | `noise_type=symmetric`, `noise_rate=0.4` |
| 模型 | `ogc-sap`, `resnet18` |
| 优化 | `optimizer=sgd`, `lr=0.03`, `n_epochs=50`, `batch_size=32`, `minibatch_size=32`, `buffer_size=500` |
| 设备 | `device=mps` |
| OGC | `ogc_loss_type=ce`, `ogc_loss_weight=0.4`, `ogc_low_conf_weight=0.3`, `ogc_buffer_penalty_coeff=2.0`, `ogc_epsilon=20.0`, `ogc_time_frame=32`, `ogc_queue_size=2048`, `ogc_warmup_epochs=10`, `ogc_lazy_update=10`, `ogc_gamma=2.0`, `ogc_q=0.7` |
| SAP | `enable_sap=1`, `sap_frequency_epochs=0`, `sap_retain_samples=400`, `sap_scale_coff=[2000.0]`, `sap_mode=['sap']`, `sap_mode_forget=['sap']`, `sap_projection_type=['Mr']`, `sap_max_samples=2000`, `sap_start_layer=[2]`, `sap_end_layer=[3]`, `sap_projection_location=['post']`, `sap_project_classifier=1` |
| AER/buffer | `sample_selection_strategy=abs`, `use_aer=1`, `alpha_sample_insertion=0.75`, `buffer_fitting_epochs=0` |

## 3. 本次最终结果

最终日志记录：

```text
Accuracy for 5 task(s): [Class-IL]: 60.3 % [Task-IL]: 89.42 %
Raw accuracy values:
Class-IL [65.10000000000001, 53.55, 46.300000000000004, 69.0, 67.55]
Task-IL  [91.75, 79.75, 87.4, 93.0, 95.19999999999999]
```

逐阶段累计平均准确率：

| 已训练 task 数 | Class-IL 平均准确率 (%) | Task-IL 平均准确率 (%) |
| --- | ---: | ---: |
| 1 | 95.20 | 95.20 |
| 2 | 75.00 | 82.82 |
| 3 | 66.73 | 88.70 |
| 4 | 63.20 | 89.30 |
| 5 | 60.30 | 89.42 |

最终逐任务 raw accuracy：

| Task | Class-IL (%) | Task-IL (%) |
| --- | ---: | ---: |
| 1 | 65.10 | 91.75 |
| 2 | 53.55 | 79.75 |
| 3 | 46.30 | 87.40 |
| 4 | 69.00 | 93.00 |
| 5 | 67.55 | 95.20 |

## 4. 与原文指标对比

原文 PDF 第 8 页 Table I 中，CIFAR-10 / symmetric noise / 40% 的结果为：

| 方法 | 原文 Class-IL final average accuracy (%) |
| --- | ---: |
| AER | 60.87 ± 0.98 |
| Ours | 64.46 ± 0.40 |

本次实验与原文 Ours 对比：

| 指标 | 本次本机结果 | 原文 Ours | 差值 |
| --- | ---: | ---: | ---: |
| Class-IL final average accuracy | 60.30 | 64.46 ± 0.40 | -4.16 |
| 原文 Ours 下界对比 | 60.30 | 64.06 | -3.76 |
| 原文 Ours 上界对比 | 60.30 | 64.86 | -4.56 |

与原文 AER 对比：

| 指标 | 本次本机结果 | 原文 AER | 差值 |
| --- | ---: | ---: | ---: |
| Class-IL final average accuracy | 60.30 | 60.87 ± 0.98 | -0.57 |

原文 Table I 未报告 Task-IL，因此本次 `Task-IL=89.42%` 只能作为本地复现实验的辅助指标，不能直接与原文主表做严格对比。

## 5. 结论

本机 CIFAR10 Symmetric 40% 实验已经完整跑完，结果文件、日志和 checkpoint 均已写出，运行过程中未发现致命异常。

本次使用 `readme_latest.md` 推荐参数得到的最终 Class-IL 为 `60.30%`，低于原文 Ours `64.46±0.40`，差值为 `-4.16` 个百分点，并且低于原文误差区间下界 `64.06%`。因此，本次单 seed 本机重跑未达到原文 CIFAR10 Symm40 指标。

从数值位置看，本次结果接近原文 AER `60.87±0.98`，但没有复现出原文 Ours 相对 AER 在 CIFAR10 Symm40 上的提升。后续若继续追原文指标，建议优先复查论文实验是否为多 seed 均值、CUDA 与 MPS 的差异、随机种子稳定性、 noisy label cache 是否与原文一致，以及本地历史调参中 `ogc_low_conf_weight=0.25`、`sap_scale_coff=3000` 等组合。

## 6. 依据文件

- 运行日志: `run_logs/cifar10_symm40_readme_latest_local_retry2_20260703_231106.log`
- 结构化结果: `data/results_cifar10_symm40_readme_latest_local_retry2_20260703_231106/class-il/seq-cifar10/ogc_sap/logs.pyd`
- Task-IL 结果: `data/results_cifar10_symm40_readme_latest_local_retry2_20260703_231106/task-il/seq-cifar10/ogc_sap/logs.pyd`
- 原文 PDF: `/Users/hunk/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_i1ps5n9ndiyk22_8782/temp/RWTemp/2026-06/9e20f478899dc29eb19741386f9343c8/Noise-Robust Class-Incremental Learning via Dynamic Gradient Constraint and Cumulative Bias Projection.pdf`
- PDF 渲染检查图: `tmp/pdfs/cifar10_symm40_paper_table_page-08.png`
