# Food101N 路径从哪里获取

日期：2026-07-14

## 结论

这些路径不是凭空有的，需要从两个来源生成：

1. **split 路径**：从 NTD 或 PuriDivER 开源仓库获取 `tasks/Food-101N/*.json`。
2. **图片路径**：从 Food101N 官方/Kaggle 下载图片数据，解压后得到本地 `images/` 目录。

本项目推荐采用 NTD/PuriDivER 的 `tasks/Food-101N` split，因为它已经包含我们需要的：

```text
52,867 train
4,741 test
101 classes
5 tasks
```

并且每条记录已经有：

```text
file_name
klass
label
verification_label
```

## 1. 获取 split 文件

推荐用 NTD 仓库：

```bash
mkdir -p external
git clone --depth 1 https://github.com/wish44165/ntd.git external/ntd
```

下载后 split 路径就是：

```text
external/ntd/tasks/Food-101N
```

里面有类似这些文件：

```text
Food-101N_train_blurry10_rand1_cls20_task0.json
Food-101N_train_blurry10_rand1_cls20_task1.json
Food-101N_train_blurry10_rand1_cls20_task2.json
Food-101N_train_blurry10_rand1_cls20_task3.json
Food-101N_train_blurry10_rand1_cls20_task4.json

Food-101N_test_rand1_cls20_task0.json
Food-101N_test_rand1_cls20_task1.json
Food-101N_test_rand1_cls20_task2.json
Food-101N_test_rand1_cls20_task3.json
Food-101N_test_rand1_cls20_task4.json
```

也有 `rand2`、`rand3`。当前项目默认先用 `rand1`，后续多 seed 或多 split 对比时再扩展到 `rand2/rand3`。

我已经用 NTD 的 `rand1` task JSON 做过 dry run：

```text
train.tsv -> 52,867 rows
test.tsv  -> 4,741 rows
classes.txt -> 101 classes
```

所以 split 文件本身不需要你手工写。

## 2. 获取图片数据

图片需要从 Food101N 官方数据源下载。

官方页面：

```text
https://kuanghuei.github.io/Food-101N/
```

Food101N 官方数据通常通过 Kaggle 分发。常见 Kaggle dataset slug 是：

```text
kuanghueilee/food-101n
```

如果使用 Kaggle CLI，流程一般是：

```bash
pip install kaggle
mkdir -p ~/.kaggle
```

然后到 Kaggle 账号页面创建 API Token，把下载到的 `kaggle.json` 放到：

```text
~/.kaggle/kaggle.json
```

并设置权限：

```bash
chmod 600 ~/.kaggle/kaggle.json
```

下载：

```bash
mkdir -p data/raw/food101n
kaggle datasets download -d kuanghueilee/food-101n -p data/raw/food101n --unzip
```

下载完成后，用下面命令找真实图片目录：

```bash
find data/raw/food101n -maxdepth 5 -type d \
  \( -iname images -o -iname train -o -iname Food-101N_release \)
```

目标是找到一个里面有类别子目录的路径，比如：

```text
data/raw/food101n/Food-101N_release/images
```

或：

```text
data/raw/food101n/images
```

这个路径就是后续命令里的：

```text
<food101n_images_dir>
```

## 3. 转成项目标准 metadata

拿到 split 目录和图片目录后，运行：

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

如果图片路径全部存在，会生成：

```text
data/Food-101N/meta/train.tsv
data/Food-101N/meta/test.tsv
data/Food-101N/meta/classes.txt
```

如果你希望把图片也整理到项目目录下，可以用软链接，节省空间：

```bash
python scripts/prepare_food101n_metadata.py \
  --output-root data/Food-101N \
  --task-dir external/ntd/tasks/Food-101N \
  --task-rand 1 \
  --train-images-dir <food101n_images_dir> \
  --test-images-dir <food101n_images_dir> \
  --strict \
  --overwrite \
  --link-images
```

这会让项目形成：

```text
data/Food-101N/
  meta/
    train.tsv
    test.tsv
    classes.txt
  images/
    train/
    test/
```

## 4. 验证

如果没有使用 `--link-images` 或 `--copy-images`，验证时继续指定原始图片目录：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --train-images-dir <food101n_images_dir> \
  --test-images-dir <food101n_images_dir> \
  --classes-file meta/classes.txt \
  --strict
```

如果使用了 `--link-images`，验证命令可以简化为：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict
```

验证通过后，才进入 1 epoch smoke test。

## 5. 当前还缺什么

现在 split 文件的获取方式已经明确：clone NTD 即可。

当前真正缺的是 Food101N 图片数据。也就是说，下一步需要你完成 Kaggle/Food101N 官方下载，或者把已经下载好的 Food101N 图片目录路径发给我。

我需要的最终路径只有一个：

```text
<food101n_images_dir>
```

例如：

```text
/Users/hunk/Downloads/Food-101N/images
```

或：

```text
/Users/hunk/Documents/datasets/Food-101N/images
```

拿到这个路径后，我可以直接执行：

```bash
python scripts/prepare_food101n_metadata.py ...
python scripts/validate_food101n_dataset.py ...
```

## 6. 注意事项

1. 不要把 Food-101 官方 25,250 test set 和 NTD/PuriDivER 的 `4,741 test` 混在一起。
2. 当前已选择 AER/NTD 接近的 `52,867 / 4,741` 协议，因此优先使用 `external/ntd/tasks/Food-101N`。
3. `test.tsv` 中的 `4,741` 是 NTD/PuriDivER task JSON 给出的 test split，不是完整 Food-101 clean test。
4. 如果后续要做 Food-101 clean test 扩展实验，可以另起一个实验协议文档，不要和当前正式复现混用。
