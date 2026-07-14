# Food101N 下一步执行路线

日期：2026-07-14

## 结论先行

接下来不要急着直接跑完整训练。Food101N 这部分最容易出问题的不是模型代码，而是数据协议。用户已确认采用论文/AER/NTD 中更接近的 `52,867 train / 4,741 test` split，而不是完整 Food101N 约 310k noisy train。

我建议下一步按这个顺序做：

1. **先整理并验证 Food101N `52,867 / 4,741` split**。
2. **把当前 `seq-food101n` 默认配置改成论文设置**。
3. **用真实数据跑 dataset validation 和 1 epoch smoke test**。
4. **确认 buffer size、backbone、epoch 后跑正式 baseline 和 Ours**。
5. **把每一步结果继续写成 `.md` 文档**。

## 第一步：确认数据协议

这是最关键的一步。

当前已经确认：

| 来源 | Food101N 线索 |
|---|---|
| Food101N 官方 | 全量约 310,009 张 noisy images；另有 52,868 train verification labels 和 4,741 val verification labels |
| AER 论文 | Seq. Food-101N 约 52,867 train，每类平均约 523，5 tasks |
| NTD 仓库 | Food-101N train 52,867，test 4,741，class 101，tasks 5，memory size 2,000，ResNet34 |
| 项目原文 | Food101N 101 类，5 tasks，ResNet34，20 epochs，batch size 32，memory follows AER |

严格复现目标已经确定：

| 选择 | 含义 | 是否推荐 |
|---|---|---|
| 使用 52,867 / 4,741 split | 更接近 AER/NTD/原文表格协议 | 已采用 |
| 扫描完整 310k Food101N train | 数据更多，但很可能不能直接对齐原文结果 | 不作为首轮正式复现 |

当前执行口径：**优先复现 52,867 train + 4,741 test/val 这套公开持续学习协议**。如果后面需要，再额外做 full Food101N 版本作为扩展实验。

## 第二步：准备真实数据

目标目录建议整理成：

```text
data/
  Food-101N/
    images/
      <class_name>/
        xxx.jpg
    meta/
      train.tsv 或 train.txt
      test.tsv 或 test.txt
      classes.txt
```

或者：

```text
data/
  Food-101N_release/
    train/
      <class_name>/
        xxx.jpg
    meta/
      train.tsv
      classes.txt

  food-101/
    images/
      <class_name>/
        xxx.jpg
    meta/
      test.txt
      classes.txt
```

需要保证：

1. train 使用 Food101N noisy class label。
2. test 使用 Food-101 clean test label，或 AER/NTD 对应 test split。
3. `classes.txt` 的类别顺序必须和 train/test label id 一致。
4. verification label 只能保存，不能当成 clean class label。

## 第三步：优先参考 NTD/PuriDivER 核对 split

建议先把 NTD 和 PuriDivER 当成“协议核对来源”，而不是直接复制代码。

重点核对：

| 项目 | 推荐值 |
|---|---|
| train samples | 52,867 |
| test samples | 4,741 |
| classes | 101 |
| tasks | 5 |
| class split | `[20, 20, 20, 20, 21]` |
| memory size | 2,000 |
| backbone | ResNet34 |
| epochs | 20 |
| batch size | 32 |

如果本地数据整理后 `validate_food101n_dataset.py` 输出样本数明显不是 52,867/4,741，就先不要跑正式实验，先回头检查 split。

## 第四步：把默认配置对齐论文

当前代码已经将默认值改成论文设置：

```text
backbone = resnet34
n_epochs = 20
```

严格复现 Food101N 时，当前推荐保持：

```text
backbone = resnet34
n_epochs = 20
batch_size = 32
minibatch_size = 32
```

这一步建议在真实数据路径确认后再改，因为配置一旦对齐，就可以直接跑标准命令。

## 第五步：先跑数据验证

真实数据准备好后，先跑：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict
```

如果 train/test 图片目录不同，则使用：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N_release \
  --train-list meta/train.tsv \
  --test-list ../food-101/meta/test.txt \
  --train-images-dir train \
  --test-images-dir ../food-101/images \
  --classes-file ../food-101/meta/classes.txt \
  --strict
```

验证目标：

1. 能找到所有图片。
2. 类别数是 101。
3. train/test label 没有越界。
4. train/test class mapping 一致。
5. 样本数接近协议目标。

## 第六步：跑 1 epoch smoke test

数据验证通过后，先跑小实验，不要直接跑正式 20 epochs。

推荐命令：

```bash
python main.py \
  --dataset seq-food101n \
  --model ogc-sap \
  --enable_sap 1 \
  --food101n_root data/Food-101N \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 \
  --n_epochs 1 \
  --batch_size 8 \
  --minibatch_size 8 \
  --buffer_size 200 \
  --noise_rate 0 \
  --num_workers 0 \
  --debug_mode 1 \
  --seed 0
```

smoke test 只看：

1. dataloader 是否能正常进入 Task 0。
2. 第一批图片 tensor shape 是否正确。
3. label 范围是否在当前 task 内。
4. 模型 forward/backward 是否正常。
5. replay buffer 是否能插入样本。

## 第七步：跑正式 Food101N 复现实验

smoke test 通过后，正式设置建议为：

```bash
python main.py \
  --dataset seq-food101n \
  --model ogc-sap \
  --enable_sap 1 \
  --food101n_root data/Food-101N \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 \
  --n_epochs 20 \
  --batch_size 32 \
  --minibatch_size 32 \
  --buffer_size 2000 \
  --noise_rate 0 \
  --ogc_queue_size 2048 \
  --ogc_warmup_epochs 10 \
  --ogc_lazy_update 10 \
  --ogc_low_conf_weight 0.2 \
  --ogc_buffer_penalty_coeff 2.0 \
  --seed 0
```

之后至少跑：

| 目的 | model |
|---|---|
| AER baseline | `er-ace-aer-abs` |
| SAP-only 消融 | `aer-sap` |
| OGC/DGC + SAP/CBP 完整方法 | `ogc-sap` |

如果时间允许，跑 3 个 seed：

```text
seed = 0, 1, 2
```

## 第八步：结果对照

最终至少对照这几个数字：

| Method | 原文 Food101N FAA |
|---|---:|
| PuriDivER.ME | 28.62 ± 0.85 |
| AER | 29.86 ± 1.18 |
| Ours | 31.62 ± 0.98 |

如果复现结果偏差很大，优先排查顺序：

1. 数据 split 是否是 52,867/4,741。
2. test 是否真的是 clean Food-101/Food101N 协议 test。
3. class order 是否一致。
4. backbone 是否是 ResNet34。
5. epochs 是否是 20。
6. buffer size 是否是 2,000。
7. 是否错误设置了 `--noise_rate > 0`。
8. transform 是否是 224 输入。

## 我建议现在马上做的事

最优先的下一步是：

```text
找到或整理 Food101N 的 52,867 train / 4,741 test split，然后用 scripts/validate_food101n_dataset.py 验证。
```

如果本机还没有数据，就先做数据准备，不要继续改模型。

如果本机已经有 Food101N 数据，但不知道是不是正确 split，就先让我检查目录结构和 metadata。我会先做只读扫描，确认样本数、类别数、文件名字段、class order，然后再决定是否需要生成中间 `train.tsv/test.tsv/classes.txt`。

## 当前代码状态与下一步

当前已经完成：

1. Food101N 默认配置已改为论文设置：`resnet34`、20 epochs。
2. 验证脚本默认检查 `52,867 train / 4,741 test`。
3. `readme_latest.md` 已同步正式验证和训练命令。

下一步如果拿到 NTD/AER split metadata，建议新增一个转换脚本，把原始 split 转成当前代码稳定支持的 `train.tsv/test.tsv/classes.txt`。

不建议现在做的事：

1. 不建议直接扫描完整 310k Food101N 作为正式复现。
2. 不建议把 verification label 当 clean label。
3. 不建议为了“用开源代码”直接复制 PuriDivER/NTD Dataset。
4. 不建议在真实 split 验证通过前跑 20 epochs 正式训练。
