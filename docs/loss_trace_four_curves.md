# Iteration 级四组 Loss 曲线

本功能为 `er-ace`、`er-ace-aer-abs`、`aer-sap` 和 `ogc-sap` 增加可选的逐样本 loss 记录。默认关闭，不改变原训练命令。

## 统计口径

每个点对应一次 `observe()`，loss 在本次 `optimizer.step()` 之前计算。为了让 current 和 memory 可比，记录器使用相同模型状态、归一化后的非增强输入、当前已见类别 logits，以及相对于 `observed_label` 的逐样本 Cross Entropy。

每个 iteration 使用当次 current batch 和当次 `buffer.get_data()` 抽到的 memory batch：

- `noisy`：current 或 old memory 中 `observed_label != true_label` 的样本。
- `new`：current 中 `observed_label == true_label` 的样本。
- `hard_old`：old memory 中标签正确样本的当前 loss 最高 20%，数量使用 `ceil(0.2 * N)`。
- `easy_old`：上述 clean old memory 的剩余样本。

Hard/Easy 每个 iteration 动态重排。memory 中 `source_task_id == current_task_id` 的样本仍保存在原始 CSV 中，但标记为 `ignored_memory_current`，不进入四条曲线。

## 训练参数

```text
--enable_loss_trace 1
--loss_trace_task 1
--loss_trace_start_epoch 5
--loss_trace_end_epoch 9
--loss_trace_dir output/loss_trace
--loss_trace_flush_every 100
--loss_trace_hard_ratio 0.2
```

所有 task/epoch 参数都是 0-based：`loss_trace_task=1` 表示第二个 task；epoch 5 到 9 对应连续横坐标 `[5, 10)`。使用 `-1` task 可记录全部任务，使用 `-1` end epoch 表示不设上限。

## 实验超参

下表把训练超参和作图/统计超参分开。这很重要：`loss_trace_hard_ratio=0.2` 只决定曲线分组，不参与训练；`alpha_sample_insertion=0.75` 则会改变 AER/ABS 的 buffer 写入。

### 论文明确给出的 Seq. CIFAR-10 / 40% symmetric 设置

| 参数 | 值 | 说明 |
|---|---:|---|
| dataset | `seq-cifar10` | 5 tasks，每 task 2 classes |
| noise type/rate | `symmetric`, `0.4` | 图 1 的 40% 对称标签噪声 |
| backbone | `resnet18` | 无预训 |
| stream batch size | `32` | 论文固定值 |
| replay minibatch size | `32` | 论文固定值 |
| epochs per task | `50` | 正式 CIFAR-10 训练协议 |
| buffer size | `500` | CIFAR-10 总 buffer 大小 |
| augmentation | random crop + horizontal flip | 仓库 `seq-cifar10/default.yaml` 已配置 |
| optimizer | `SGD` | 当前仓库的复现协议；论文附录对主训练只列出了最佳 lr，未完整列出 optimizer 配置 |
| learning rate (ER baseline) | `0.1` | 论文附录 CIFAR-10, sym 40%, ER 最佳值 |
| learning rate (AER/ABS OURs) | `0.03` | 论文附录 CIFAR-10, sym 40%, OURs 最佳值 |
| repetitions | `5` | 正式结果使用 5 次实验；单条过程曲线可先固定 `seed=0` |

论文没有单独公开图 1 的随机种子、曲线平滑窗口，也没在该超参表明说 momentum/weight decay/scheduler。为使当前仓库的运行完全可复现，命令显式固定项目默认值：`optim_mom=0`、`optim_wd=0`、`optim_nesterov=0`、不使用 scheduler。这四项是“仓库协议”，不应写成论文明确报告的值。

### 四线图统计超参

| 参数 | 值 | 作用 |
|---|---:|---|
| traced task | `1` | 0-based 的第二个 task |
| traced epochs | `5..9` | 横轴连续显示 `[5, 10)` |
| hard old ratio | `0.2` | 当次抽到的 clean old memory 中 loss 最高 20% |
| easy old ratio | `0.8` | 剩余 clean old memory，由上一项的补集得到 |
| loss reduction | per-sample CE, then group mean | 先记录每个样本 loss，再对当次 iteration 各组取均值 |
| loss timing | before `optimizer.step()` | 四组使用同一个参数状态 |
| plot smoothing | `5` (建议) | 只影响显示，CSV 保留原始值；需原始曲线用 `1` |

## 标准 Replay（图 1 左图训练口径）

这里必须使用真正的 `er-ace` + reservoir replay。`er-ace-aer-abs --use_aer 0` 仍然会进行高损失样本过滤，不是纯 ER-ACE 左图基线。

```bash
python main.py \
  --dataset seq-cifar10 \
  --model er-ace \
  --backbone resnet18 \
  --noise_rate 0.4 \
  --noise_type symm \
  --n_epochs 50 \
  --batch_size 32 \
  --minibatch_size 32 \
  --buffer_size 500 \
  --optimizer sgd \
  --lr 0.1 \
  --optim_mom 0 \
  --optim_wd 0 \
  --optim_nesterov 0 \
  --use_custom_classifier 1 \
  --seed 0 \
  --stop_after 2 \
  --enable_loss_trace 1 \
  --loss_trace_task 1 \
  --loss_trace_start_epoch 5 \
  --loss_trace_end_epoch 9 \
  --loss_trace_hard_ratio 0.2 \
  --loss_trace_dir output/loss_trace
```

`--stop_after 2` 表示训完 task 0 和 task 1 后停止，足以生成第二个 task 的曲线，不会改变前两个 task 的训练。

## AER/ABS 方法对照命令

如果要看交替 replay 和 ABS 下的同一组四条曲线，使用论文 OURs 的学习率与方法超参：

```bash
python main.py \
  --dataset seq-cifar10 \
  --model er-ace-aer-abs \
  --backbone resnet18 \
  --noise_rate 0.4 \
  --noise_type symm \
  --n_epochs 50 \
  --batch_size 32 \
  --minibatch_size 32 \
  --buffer_size 500 \
  --optimizer sgd \
  --lr 0.03 \
  --optim_mom 0 \
  --optim_wd 0 \
  --optim_nesterov 0 \
  --use_aer 1 \
  --sample_selection_strategy abs \
  --alpha_sample_insertion 0.75 \
  --buffer_fitting_epochs 0 \
  --seed 0 \
  --stop_after 2 \
  --enable_loss_trace 1 \
  --loss_trace_task 1 \
  --loss_trace_start_epoch 5 \
  --loss_trace_end_epoch 9 \
  --loss_trace_hard_ratio 0.2 \
  --loss_trace_dir output/loss_trace
```

`alpha_sample_insertion=0.75` 表示丢弃 current batch 中 loss 最高的 75%，只把低 loss 25% 作为 buffer 写入候选。它与四线图中 hard/easy old 的 20%/80% 完全独立。

`ogc-sap` 和 `aer-sap` 可追加同样的 trace 参数，但其 OGC/SAP 超参必须另外固定，不能把上面的纯 ER-ACE 命令直接当成 OGC-SAP 的完整实验配置。

## 输出文件

每次运行创建独立目录：

```text
output/loss_trace/<dataset>_<model>_<run-id>/
├── metadata.json
├── sample_losses.csv
└── iteration_curves.csv
```

`sample_losses.csv` 每行是一个样本在某个 iteration 的记录，包含 task/epoch/iteration、sample id、buffer slot、来源 task、observed/true class、noisy 标识、loss 和动态 group。

`iteration_curves.csv` 每行是一个 iteration，包含四组 mean loss 和四组样本数。某组没有样本时 loss 为 `NaN`、count 为 0，不会伪造为零 loss。

## 画图

项目的 `matplotlib` 是可选依赖；缺少时先安装：

```bash
pip install pandas matplotlib
```

然后运行：

```bash
python scripts/plot_loss_trace.py \
  output/loss_trace/<run-directory> \
  --smooth-window 5
```

默认输出 `<run-directory>/four_loss_curves.png`。`smooth-window=1` 绘制原始 iteration 曲线；大于 1 只对显示结果做居中滚动平均，原始 CSV 不受影响。
