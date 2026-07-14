# Food101N Split 决策与项目同步记录

日期：2026-07-14

## 本轮决策

用户已确认：Food101N 正式复现采用论文/AER/NTD 接近的 `52,867 train / 4,741 test` split。

这意味着：

1. 首轮正式复现不直接扫描完整约 310k Food101N train images。
2. Food101N 数据必须通过 metadata 明确限定样本集合。
3. 推荐把目标 split 整理成稳定的 `train.tsv`、`test.tsv`、`classes.txt`。
4. 验证脚本默认检查 train/test 样本数是否分别为 `52,867` 和 `4,741`。
5. 训练配置对齐论文/AER/NTD 风格：ResNet34、20 epochs、batch size 32、buffer size 2000、`noise_rate=0`。

## 已同步的项目改动

### 1. 默认配置已改为论文设置

文件：

```text
datasets/configs/seq-food101n/default.yaml
```

已从：

```yaml
backbone: resnet18
n_epochs: 50
```

改为：

```yaml
backbone: resnet34
n_epochs: 20
```

保留：

```yaml
batch_size: 32
minibatch_size: 32
```

### 2. 验证脚本默认检查目标 split 样本数

文件：

```text
scripts/validate_food101n_dataset.py
```

新增默认目标：

```python
EXPECTED_TRAIN_RECORDS = 52867
EXPECTED_TEST_RECORDS = 4741
```

新增参数：

```bash
--expected-train-records
--expected-test-records
```

默认会检查：

```text
train records == 52,867
test records  == 4,741
```

如果只是临时检查非正式 split，可以关闭样本数检查：

```bash
--expected-train-records 0 --expected-test-records 0
```

### 3. `readme_latest.md` 已更新 Food101N 命令模板

当前推荐验证命令：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict
```

当前推荐 smoke test：

```bash
python main.py --dataset seq-food101n --model ogc-sap --enable_sap 1 \
  --food101n_root data/Food-101N \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 --n_epochs 1 --batch_size 8 --minibatch_size 8 \
  --lr 0.03 --buffer_size 500 --num_workers 0 --debug_mode 1 \
  --noise_rate 0 --seed 0
```

当前推荐正式训练公共参数：

```bash
COMMON_ARGS="--dataset seq-food101n \
  --food101n_root data/Food-101N \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 \
  --n_epochs 20 \
  --batch_size 32 \
  --minibatch_size 32 \
  --lr 0.03 \
  --buffer_size 2000 \
  --num_workers 4 \
  --noise_rate 0"
```

### 4. AGENTS 项目记忆已更新

文件：

```text
AGENTS.md
```

已记录：

1. Food101N 采用 `52,867 train / 4,741 test` split。
2. 默认 backbone/epoch 为 `resnet34`、20。
3. 验证脚本默认检查 `52,867 / 4,741`。
4. 真实数据路径仍待准备和验证。

### 5. 既有 Food101N 文档已同步

已同步更新：

```text
docs/food101n/06_paper_food101n_protocol_and_code_review.md
docs/food101n/09_food101n_next_steps.md
```

主要变化：

1. 从“推荐/待确认 split”改为“已确认采用 52,867 / 4,741 split”。
2. 从“默认 resnet18/50 未对齐”改为“默认 resnet34/20 已对齐”。
3. 正式命令中的 buffer size 采用 2000。

## 仍需注意的地方

`buffer_size=2000` 是按 AER/NTD 风格采用的项目复现协议。项目原文主文只说 Food101N memory follows AER，没有在主文单独明示这个数值。因此后续报告应表述为：

```text
本复现按 AER/NTD 风格采用 Food101N memory size 2000。
```

不要写成：

```text
原文主文明示 Food101N buffer size 为 2000。
```

## 下一步

下一步应准备或定位真实 Food101N split 文件：

```text
data/Food-101N/meta/train.tsv
data/Food-101N/meta/test.tsv
data/Food-101N/meta/classes.txt
data/Food-101N/images/
```

然后先运行：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict
```

验证通过后，再运行 1 epoch smoke test。不要在 split 验证通过前直接跑 20 epochs 正式训练。
