# Food101N 本地扫描与 Metadata 准备工具

日期：2026-07-14

## 本轮做了什么

用户要求继续 Food101N 的下一步。根据前面已经确认的复现协议，本轮先做两件事：

1. 只读扫描本机是否已有 Food101N/Food-101 数据或 split 文件。
2. 新增 metadata 准备脚本，把未来拿到的原始 split 转成项目标准格式。

## 本地数据扫描结果

已扫描这些常见位置：

```text
mammoth_code/data/
/Users/hunk/Documents/
/Users/hunk/Downloads/
/Users/hunk/Desktop/
当前仓库内部文件
```

搜索关键词包括：

```text
Food101N
Food-101N
Food-101
food101n
food-101n
verified_train
verified_test
train.tsv
test.tsv
classes.txt
```

结果：

1. 当前仓库 `data/` 下只发现 CIFAR10、已有实验结果和 noisy label cache。
2. 未发现真实 Food101N/Food-101 图片目录。
3. 本机项目内未发现 AER/NTD/Food101N 对应的 `52,867 train / 4,741 test` split 文件。
4. 后续已确认 NTD/PuriDivER 仓库提供 `tasks/Food-101N` split JSON，可通过 clone 仓库获取；当前真正缺的是 Food101N 图片数据。
5. 因此当前还不能运行真实 Food101N validation、batch check 或训练。

## 新增脚本

新增文件：

```text
scripts/prepare_food101n_metadata.py
```

作用：把外部来源的 Food101N/Food-101 split metadata 转成项目推荐格式：

```text
data/Food-101N/
  meta/
    train.tsv
    test.tsv
    classes.txt
  images/
```

其中 `images/` 可以不实际复制图片。如果图片已经放在别处，只生成 `meta/*.tsv` 也可以，后续验证/训练时通过 `--train-images-dir` 和 `--test-images-dir` 指向原始图片目录。

## 准备脚本支持什么输入

`prepare_food101n_metadata.py` 复用了当前验证脚本的数据解析逻辑，支持：

| 输入类型 | 支持情况 |
|---|---|
| JSON list | 支持 |
| JSON dict with `annotations/samples/data/images/items` | 支持 |
| Food-101 style `{class_name: [image_id, ...]}` JSON | 支持 |
| TXT/CSV/TSV | 支持 |
| class-folder scan | 支持，但正式复现不建议只靠扫描完整目录 |
| classes file | 支持 `.txt` 或 `.json` |

支持字段：

| 类型 | 字段名 |
|---|---|
| 图片路径 | `file_name`, `filepath`, `file_path`, `path`, `image`, `img`, `sample_key` |
| 类名 | `klass`, `class`, `class_name`, `category`, `category_name`, `label_name` |
| 类别 id | `label`, `target`, `class_idx`, `class_id`, `category_id` |
| verification | `verification_label`, `verified`, `is_verified` |

## 推荐用法

如果使用 NTD/PuriDivER 风格 task split，先获取 task JSON：

```bash
mkdir -p external
git clone --depth 1 https://github.com/wish44165/ntd.git external/ntd
```

然后运行：

```bash
python scripts/prepare_food101n_metadata.py \
  --output-root data/Food-101N \
  --task-dir external/ntd/tasks/Food-101N \
  --task-rand 1 \
  --train-images-dir <food101n_images_dir> \
  --test-images-dir <food101n_images_dir> \
  --strict \
  --overwrite
```

如果手头是其他单文件 split metadata，也仍可使用：

```bash
python scripts/prepare_food101n_metadata.py \
  --output-root data/Food-101N \
  --train-list <source_train_split> \
  --test-list <source_test_split> \
  --train-images-dir <source_train_images> \
  --test-images-dir <source_test_images> \
  --classes-file <source_classes_file> \
  --strict \
  --overwrite
```

脚本默认检查：

```text
train records == 52,867
test records  == 4,741
classes       == 101
```

如果计数不符合，会停止，避免误把完整 310k Food101N 当作正式复现 split。

临时非正式检查可加：

```bash
--allow-count-mismatch
```

但正式复现不要加这个参数。

## 是否复制图片

默认不复制图片，只写 metadata。

如果希望整理成完整标准目录，可以加：

```bash
--link-images
```

这会把选中 split 的图片软链接到：

```text
data/Food-101N/images/train/...
data/Food-101N/images/test/...
```

也可以加：

```bash
--copy-images
```

但复制 57k 张左右图片会占用额外磁盘空间，不作为首选。

## 转换后的验证命令

如果没有复制/链接图片，脚本会打印类似下面的验证命令：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --train-images-dir <source_train_images> \
  --test-images-dir <source_test_images> \
  --classes-file meta/classes.txt \
  --strict
```

如果使用 `--link-images` 或 `--copy-images`，验证命令会变成：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict
```

## 下一步需要用户提供什么

当前最缺的是原始数据路径。下一步请提供以下任一组路径：

### 最理想

```text
source_train_split
source_test_split
source_train_images
source_test_images
source_classes_file
```

### 如果只有图片目录

也可以先提供：

```text
Food101N noisy train 图片目录
Food-101 clean test 图片目录
类别名文件或 meta/classes.txt
```

但如果没有 split 文件，只靠扫描目录很可能无法保证 `52,867 / 4,741` 协议。

## 当前状态

代码层面已准备好：

1. Food101N dataset loader 已有。
2. Food101N validation script 已有。
3. Food101N metadata preparation script 已有。
4. 默认配置已对齐 `resnet34`、20 epochs。
5. `readme_latest.md` 和 `AGENTS.md` 已同步。

数据层面仍阻塞：

```text
还没有找到真实 Food101N 图片数据；52,867/4,741 split metadata 可从 NTD/PuriDivER 的 `tasks/Food-101N` 获取。
```

拿到路径后，下一步就是运行 `prepare_food101n_metadata.py`，再运行 `validate_food101n_dataset.py`。
