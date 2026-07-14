# Food101N 除图片下载外的准备完成记录

日期：2026-07-14

## 结论

除真实 Food101N 图片下载外，当前 Food101N 复现前置工作已经基本完成。

已完成：

1. 获取 NTD 的 `tasks/Food-101N` split。
2. 用 `rand1` split 生成本项目标准 metadata。
3. 验证 metadata 样本数、类别数、label 范围和 task 分布。
4. 更新准备脚本、验证脚本、README 和项目记忆。
5. 确认下一步只缺 Food101N 图片目录。

当前还缺：

```text
data/Food-101N/images/
```

也就是 Food101N 图片本体。下载图片后即可进入严格图片路径验证和 1 epoch smoke test。

## 已获取 split

已将 NTD 仓库 clone 到：

```text
external/ntd
```

其中 Food101N task split 位于：

```text
external/ntd/tasks/Food-101N
```

`external/` 已加入 `.gitignore`，不会把第三方仓库混进项目提交。

## 已生成 metadata

已运行：

```bash
python scripts/prepare_food101n_metadata.py \
  --output-root data/Food-101N \
  --task-dir external/ntd/tasks/Food-101N \
  --task-rand 1 \
  --train-images-dir data/Food-101N/images \
  --test-images-dir data/Food-101N/images \
  --overwrite
```

已生成：

```text
data/Food-101N/meta/train.tsv
data/Food-101N/meta/test.tsv
data/Food-101N/meta/classes.txt
```

注意：`data/` 在 `.gitignore` 中，因此这些 metadata 是本机工作文件，不会默认进入 Git 提交。

## Metadata 检查结果

离线验证命令：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --skip-image-path-check
```

结果：

| 项目 | train | test |
|---|---:|---:|
| records | 52,867 | 4,741 |
| classes | 101 | 101 |
| label range | 0..100 | 0..100 |
| verification records | 52,867 | 4,741 |

按当前 class-incremental label range `[20, 20, 20, 20, 21]` 的样本分布：

| Task | train records | test records |
|---:|---:|---:|
| 0 | 10,467 | 946 |
| 1 | 10,455 | 921 |
| 2 | 10,454 | 969 |
| 3 | 10,442 | 933 |
| 4 | 11,049 | 972 |

verification label 分布：

| split | verification=1 | verification=0 |
|---|---:|---:|
| train | 43,121 | 9,746 |
| test | 3,824 | 917 |

类别顺序来自 NTD/PuriDivER task JSON 的 label mapping。前 10 类为：

```text
strawberry_shortcake
poutine
cheesecake
waffles
falafel
samosa
hot_dog
red_velvet_cake
ravioli
gyoza
```

## 已更新脚本

### `scripts/prepare_food101n_metadata.py`

现在支持直接读取 NTD/PuriDivER task 目录：

```bash
--task-dir external/ntd/tasks/Food-101N
--task-rand 1
```

并会自动合并 5 个 train task JSON 和 5 个 test task JSON。

如果指定 `--copy-images` 或 `--link-images`，但图片不存在，脚本会报错，避免生成坏链接或不完整图片目录。

### `scripts/validate_food101n_dataset.py`

现在如果使用：

```bash
--skip-image-path-check
```

输出会明确显示：

```text
missing image paths: <skipped>
```

避免把跳过检查误读为图片都存在。

## 当前正确的下一步

下载或放置 Food101N 图片到：

```text
data/Food-101N/images/
```

图片目录下应能匹配 `train.tsv/test.tsv` 中的相对路径，例如：

```text
data/Food-101N/images/strawberry_shortcake/...
data/Food-101N/images/poutine/...
data/Food-101N/images/cheesecake/...
```

图片到位后运行严格验证：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict
```

如果通过，再跑 Mammoth loader 检查：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict \
  --check-import \
  --check-batch
```

最后再跑 1 epoch smoke test。

## 重要协议提醒

当前正式协议采用 NTD/PuriDivER 的 Food101N `52,867 train / 4,741 test` split。

不要混用：

```text
Food-101 官方 25,250 clean test
```

如果以后要做 Food-101 clean test 扩展实验，应另起实验协议和文档。
