# 向日葵远程服务器运行 Food101N 实验完整流程

## 0. 目标和约束

本流程用于在向日葵远程服务器上，从项目上传、虚拟环境准备、Food101N 数据准备、metadata 生成、数据验证，到 smoke test 和正式训练，一步一步跑通 Food101N 实验。

服务器当前条件：

- Miniconda 已安装在 `/home/hnh/software/miniconda3`。
- 已有可跑 CIFAR10/100 的虚拟环境，例如 `/home/hnh/conda_envs/nrgp-mammoth`。
- 所有操作都放在 `/home/hnh` 下。
- 不建议使用 `sudo`，因为本任务不需要写系统目录；若确实缺系统工具，优先用 conda/pip 或 Python 标准库解决。

推荐最终目录结构：

```text
/home/hnh/
├── conda_envs/
│   ├── nrgp-mammoth/
│   └── nrgp-food101n/              # 只有重建环境时才需要
├── software/
│   └── miniconda3/
├── mammoth_code_food101n/          # Food101N 专用代码目录
│   ├── external/
│   │   └── ntd/
│   ├── data/
│   │   └── Food-101N/
│   │       ├── meta/
│   │       │   ├── train.tsv
│   │       │   ├── test.tsv
│   │       │   └── classes.txt
│   │       └── images/             # 推荐放 57,608 张图片的软链接
│   └── ...
├── datasets/
│   └── Food-101N/
│       └── raw/                    # 原始 Food101N 解压目录
├── food101n_runs/                  # 训练日志
└── tmp/                            # 临时上传压缩包
```

## 1. 进入服务器工作目录

打开向日葵远程桌面里的 WSL/bash 终端后，先执行：

```bash
cd /home/hnh
pwd
ls
```

确认输出路径是：

```text
/home/hnh
```

检查 GPU 和磁盘：

```bash
nvidia-smi
df -h /home/hnh
```

Food101N 图片包约十几 GB，解压和保留中间文件时建议 `/home/hnh` 至少预留 40GB 到 60GB 空间。

## 2. 上传或拉取项目代码

### 方案 A：推荐用 GitHub 拉取

本项目当前 GitHub remote 是：

```text
git@github.com:nuohanhe69-pixel/noise-class-incremental-reproduction.git
```

在本地 Mac 上先确认 Food101N 相关代码已经提交并推送：

```bash
cd "/Users/hunk/Documents/deep learning/噪声类增量项目复现/mammoth_code"
git status
git add AGENTS.md .gitignore datasets/seq_food101n.py datasets/configs/seq-food101n/default.yaml \
  scripts/validate_food101n_dataset.py scripts/prepare_food101n_metadata.py \
  readme_latest.md docs/food101n
git commit -m "Add Food101N dataset support and server runbook"
git push github HEAD
```

如果已经提交过，则只需要：

```bash
git push github HEAD
```

然后在服务器 `/home/hnh` 下拉取：

```bash
cd /home/hnh
git clone git@github.com:nuohanhe69-pixel/noise-class-incremental-reproduction.git mammoth_code_food101n
cd /home/hnh/mammoth_code_food101n
git status
```

如果服务器没有配置 GitHub SSH key，可以改用 HTTPS：

```bash
cd /home/hnh
git clone https://github.com/nuohanhe69-pixel/noise-class-incremental-reproduction.git mammoth_code_food101n
cd /home/hnh/mammoth_code_food101n
```

如果仓库是私有仓库，HTTPS 会要求 GitHub token。不要把 token 写进日志文件。

### 方案 B：用压缩包上传

如果 GitHub 暂时不好用，可以在本地把项目压缩后，通过向日葵文件传输上传到服务器的 `/home/hnh/tmp/`。

本地压缩时不要带大数据、缓存和 checkpoint：

```bash
cd "/Users/hunk/Documents/deep learning/噪声类增量项目复现"
tar --exclude='mammoth_code/.git' \
    --exclude='mammoth_code/data' \
    --exclude='mammoth_code/checkpoints' \
    --exclude='mammoth_code/external' \
    --exclude='mammoth_code/__pycache__' \
    -czf mammoth_code_food101n.tar.gz mammoth_code
```

上传到服务器 `/home/hnh/tmp/mammoth_code_food101n.tar.gz` 后，在服务器解压：

```bash
cd /home/hnh
mkdir -p mammoth_code_food101n
tar -xzf /home/hnh/tmp/mammoth_code_food101n.tar.gz -C /home/hnh/mammoth_code_food101n --strip-components=1
cd /home/hnh/mammoth_code_food101n
```

## 3. 激活已有 CIFAR 虚拟环境

优先复用已有 CIFAR10/100 环境。

```bash
source /home/hnh/software/miniconda3/etc/profile.d/conda.sh
conda activate /home/hnh/conda_envs/nrgp-mammoth
cd /home/hnh/mammoth_code_food101n
```

验证 Python、PyTorch、torchvision 和 CUDA：

```bash
python --version
python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果 `cuda available: True`，说明这个环境可以先继续用。

安装或补齐本项目依赖：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

再次验证项目能导入：

```bash
python - <<'PY'
import torch
import torchvision
from datasets.seq_food101n import SequentialFood101N
print("Food101N dataset import ok:", SequentialFood101N.NAME)
print("cuda available:", torch.cuda.is_available())
PY
```

## 4. 如果已有环境不可用，再重建虚拟环境

只有当已有环境出现 PyTorch/CUDA 不可用、包冲突严重、或无法导入项目时，才建议重建。

```bash
source /home/hnh/software/miniconda3/etc/profile.d/conda.sh
conda create -p /home/hnh/conda_envs/nrgp-food101n python=3.10 -y
conda activate /home/hnh/conda_envs/nrgp-food101n
cd /home/hnh/mammoth_code_food101n
```

先安装 GPU 版 PyTorch。以官方 PyTorch 安装页为准，选择：

```text
OS: Linux
Package: Pip
Language: Python
Compute Platform: 按服务器驱动选择 CUDA 版本
```

示例，如果服务器驱动支持 CUDA 12.8：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

如果驱动较旧，则按 PyTorch 官方页面选择 `cu126` 或 `cu118`。安装后再执行：

```bash
pip install -r requirements.txt
```

验证：

```bash
python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果这里 `cuda available` 仍然是 `False`，先不要跑 Food101N 正式实验，应先解决 CUDA/PyTorch 安装问题。

## 5. 获取 Food101N 增量 split

当前项目采用 NTD/PuriDivER 风格的 `52,867 train / 4,741 test` split。服务器上需要获取 NTD 的 task JSON。

```bash
cd /home/hnh/mammoth_code_food101n
mkdir -p external
git clone --depth 1 https://github.com/wish44165/ntd.git external/ntd
```

检查 Food101N task 文件：

```bash
ls external/ntd/tasks/Food-101N | head
```

验证 rand1 的数量：

```bash
python - <<'PY'
import json
from pathlib import Path
base = Path("external/ntd/tasks/Food-101N")
for split in ["train", "test"]:
    total = 0
    for i in range(5):
        if split == "train":
            name = f"Food-101N_train_blurry10_rand1_cls20_task{i}.json"
        else:
            name = f"Food-101N_test_rand1_cls20_task{i}.json"
        n = len(json.loads((base / name).read_text()))
        print(split, i, n)
        total += n
    print(split, "total", total)
PY
```

期望结果：

```text
train total 52867
test total 4741
```

## 6. 上传或下载 Food101N 图片

Food101N 图片是大文件，不建议放进 Git 仓库。推荐上传或下载到：

```text
/home/hnh/datasets/Food-101N/raw/
```

### 方案 A：本地下载后上传

在本地下载 Food-101N 压缩包后，通过向日葵文件传输上传到：

```text
/home/hnh/tmp/
```

例如上传后文件是：

```text
/home/hnh/tmp/Food-101N_release.zip
```

服务器解压：

```bash
mkdir -p /home/hnh/datasets/Food-101N/raw
python -m zipfile -e /home/hnh/tmp/Food-101N_release.zip /home/hnh/datasets/Food-101N/raw
```

如果是 `.tar.gz`：

```bash
mkdir -p /home/hnh/datasets/Food-101N/raw
tar -xzf /home/hnh/tmp/Food-101N_release.tar.gz -C /home/hnh/datasets/Food-101N/raw
```

### 方案 B：服务器直接下载

如果你使用 Kaggle，可以在当前虚拟环境里安装 Kaggle CLI：

```bash
pip install kaggle
```

注意：当前服务器终端用户可能是 `xjl`，但本任务只允许在 `/home/hnh` 下操作。因此不要把凭证放到默认的 `~/.kaggle`，而是放到 `/home/hnh/.kaggle`，并在每次使用 Kaggle 前设置：

```bash
export KAGGLE_CONFIG_DIR=/home/hnh/.kaggle
```

#### 推荐：使用旧版 `kaggle.json`

如果当前环境是 Python 3.10，`pip install kaggle` 可能安装到 `kaggle 1.7.x`。这个版本最稳的是使用 Kaggle 页面下方的“旧版 API 凭证 kaggle.json”，而不是新版 API 令牌。

在本机 Kaggle 设置页下载 `kaggle.json`，通过向日葵上传到服务器 Windows 下载目录后，在 WSL 中执行：

```bash
mkdir -p /home/hnh/.kaggle /home/hnh/tmp
find /mnt/c/Users -maxdepth 4 -iname 'kaggle.json' 2>/dev/null
```

假设找到的是：

```text
/mnt/c/Users/a/Downloads/kaggle.json
```

则复制到 `/home/hnh` 下：

```bash
cp /mnt/c/Users/a/Downloads/kaggle.json /home/hnh/.kaggle/kaggle.json
chmod 600 /home/hnh/.kaggle/kaggle.json
export KAGGLE_CONFIG_DIR=/home/hnh/.kaggle
```

测试：

```bash
kaggle --version
kaggle datasets list -s food-101n | head
```

#### 可选：使用新版 API 令牌

Kaggle 设置页上方的“API 令牌”需要 `Kaggle CLI >= 1.8.0` 或 `KaggleHub >= 0.4.1`。如果服务器当前 `kaggle --version` 是 `1.7.x`，新版令牌可能不能直接用。

可以先尝试升级：

```bash
pip install -U "kaggle>=1.8.0"
kaggle --version
```

如果升级后版本仍低于 1.8，或因为 Python 版本不支持而升级失败，请改用上面的旧版 `kaggle.json`。

如果确实已经是 `kaggle >= 1.8.0`，可以把新版 API token 放到：

```bash
mkdir -p /home/hnh/.kaggle
nano /home/hnh/.kaggle/access_token
chmod 600 /home/hnh/.kaggle/access_token
export KAGGLE_CONFIG_DIR=/home/hnh/.kaggle
```

然后测试：

```bash
kaggle datasets list -s food-101n | head
```

#### 下载数据

Kaggle 凭证测试通过后：

```bash
mkdir -p /home/hnh/tmp /home/hnh/datasets/Food-101N/raw
export KAGGLE_CONFIG_DIR=/home/hnh/.kaggle
kaggle datasets download -d kuanghueilee/food-101n -p /home/hnh/tmp
python -m zipfile -e /home/hnh/tmp/food-101n.zip /home/hnh/datasets/Food-101N/raw
```

如果 `kaggle datasets list` 或 `kaggle datasets download` 仍然连接失败，说明服务器命令行也访问不了 Kaggle。这时不要继续折腾服务器浏览器，改为在本机下载 Food101N zip，然后通过向日葵文件传输上传到 `/home/hnh/tmp/`，再在服务器解压。

## 7. 找到真正的图片根目录

NTD JSON 里的图片路径形如：

```text
caprese_salad/669be1415f3363cce83bd0b1873c6a1a.jpg
```

所以图片根目录必须是“下面直接有 101 个类别文件夹”的目录，例如：

```text
/home/hnh/datasets/Food-101N/raw/.../images/
```

或者：

```text
/home/hnh/datasets/Food-101N/raw/.../train/
```

查找图片目录：

```bash
find /home/hnh/datasets/Food-101N/raw -maxdepth 4 -type d | head -80
find /home/hnh/datasets/Food-101N/raw -type f | grep -Ei '\.(jpg|jpeg|png)$' | head
```

假设找到的图片根目录是：

```text
/home/hnh/datasets/Food-101N/raw/Food-101N/images
```

就设置：

```bash
export FOOD_IMAGES=/home/hnh/datasets/Food-101N/raw/Food-101N/images
```

如果你的真实路径不同，把上面变量改成真实路径。

检查类别目录：

```bash
echo "$FOOD_IMAGES"
ls "$FOOD_IMAGES" | head
test -d "$FOOD_IMAGES/apple_pie" && echo "class folder ok"
```

如果 `apple_pie` 不存在，说明 `FOOD_IMAGES` 可能还不是正确根目录。

## 8. 生成 Food101N metadata 和软链接

推荐用软链接方式：真实图片保留在 `/home/hnh/datasets/Food-101N/raw/...`，项目只在 `data/Food-101N/images` 下建立 57,608 张实验图片的链接。

```bash
cd /home/hnh/mammoth_code_food101n

python scripts/prepare_food101n_metadata.py \
  --output-root data/Food-101N \
  --task-dir external/ntd/tasks/Food-101N \
  --task-rand 1 \
  --train-images-dir "$FOOD_IMAGES" \
  --test-images-dir "$FOOD_IMAGES" \
  --link-images \
  --overwrite
```

期望输出里应看到：

```text
wrote: data/Food-101N/meta/train.tsv (52867 rows)
wrote: data/Food-101N/meta/test.tsv (4741 rows)
wrote: data/Food-101N/meta/classes.txt (101 classes)
missing images: train=0, test=0
```

如果 missing images 不是 0：

1. 先检查 `FOOD_IMAGES` 是否设置错。
2. 再检查下载的是否是完整 Food101N 图片包。
3. 不要继续正式训练。

如果软链接失败，可以改用复制：

```bash
python scripts/prepare_food101n_metadata.py \
  --output-root data/Food-101N \
  --task-dir external/ntd/tasks/Food-101N \
  --task-rand 1 \
  --train-images-dir "$FOOD_IMAGES" \
  --test-images-dir "$FOOD_IMAGES" \
  --copy-images \
  --overwrite
```

复制会占更多磁盘空间。

## 9. 验证 Food101N 数据

先做严格 metadata 和路径检查：

```bash
cd /home/hnh/mammoth_code_food101n

python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict
```

期望看到：

```text
[train] records: 52867
[test] records: 4741
[train] missing image paths: 0
[test] missing image paths: 0
[food101n] validation passed
```

再检查 Mammoth loader 能否构造：

```bash
python scripts/validate_food101n_dataset.py \
  --root data/Food-101N \
  --train-list meta/train.tsv \
  --test-list meta/test.tsv \
  --images-dir images \
  --classes-file meta/classes.txt \
  --strict \
  --check-import
```

最后真实读取一个 train/test batch：

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

如果 `--check-batch` 失败，多半是图片路径、坏图、Pillow 读取、或显存/worker 问题。

## 10. 运行 Food101N debug smoke

先用 1 epoch、小 batch、`num_workers=0` 跑通调用链。这个结果不作为论文结果，只用于确认环境和数据没问题。

```bash
cd /home/hnh/mammoth_code_food101n
mkdir -p /home/hnh/food101n_runs/smoke
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export FOOD101N_ROOT=/home/hnh/mammoth_code_food101n/data/Food-101N

nohup python -u main.py --dataset seq-food101n --model ogc-sap --enable_sap 1 \
  --food101n_root "$FOOD101N_ROOT" \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 --n_epochs 1 --batch_size 8 --minibatch_size 8 \
  --lr 0.03 --buffer_size 500 --num_workers 0 --debug_mode 1 \
  --noise_rate 0 --seed 0 \
  > /home/hnh/food101n_runs/smoke/food101n_ogc_sap_smoke_seed0.log 2>&1 &
```

查看日志：

```bash
tail -f /home/hnh/food101n_runs/smoke/food101n_ogc_sap_smoke_seed0.log
```

另开一个终端看 GPU：

```bash
nvidia-smi -l 2
```

smoke 成功后再进入正式训练。

## 11. 运行 Food101N 正式主实验

正式主实验先跑 OGC+SAP。

```bash
cd /home/hnh/mammoth_code_food101n
mkdir -p /home/hnh/food101n_runs/main
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export FOOD101N_ROOT=/home/hnh/mammoth_code_food101n/data/Food-101N

COMMON_ARGS="--dataset seq-food101n \
  --food101n_root ${FOOD101N_ROOT} \
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

nohup python -u main.py --model ogc-sap --enable_sap 1 \
  --sap_scale_coff 5000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 2000 \
  --savecheck last --seed 0 ${COMMON_ARGS} \
  > /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0.log 2>&1 &
```

看日志：

```bash
tail -f /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0.log
```

查看后台进程：

```bash
ps -u "$USER" -f | grep main.py | grep -v grep
```

停止某个训练：

```bash
kill <PID>
```

## 12. 多 seed 实验

主实验 seed 0 跑通后，可以按 seed 0/1/2 跑。

```bash
cd /home/hnh/mammoth_code_food101n
mkdir -p /home/hnh/food101n_runs/main
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export FOOD101N_ROOT=/home/hnh/mammoth_code_food101n/data/Food-101N

COMMON_ARGS="--dataset seq-food101n \
  --food101n_root ${FOOD101N_ROOT} \
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

for SEED in 0 1 2; do
  nohup python -u main.py --model ogc-sap --enable_sap 1 \
    --sap_scale_coff 5000 \
    --ogc_loss_weight 0.3 \
    --ogc_low_conf_weight 0.3 \
    --ogc_buffer_penalty_coeff 2.0 \
    --sap_retain_samples 2000 \
    --savecheck last --seed ${SEED} ${COMMON_ARGS} \
    > /home/hnh/food101n_runs/main/food101n_ogc_sap_seed${SEED}.log 2>&1 &
done
```

如果只有一张 GPU，不建议三个 seed 同时跑。更稳妥的方法是一个 seed 跑完再开下一个。

## 13. 显存不足时怎么处理

如果日志里出现 `CUDA out of memory`，先不要改正式实验参数，先用小参数确认能跑：

```bash
--batch_size 16 --minibatch_size 16 --num_workers 2
```

如果还不行：

```bash
--batch_size 8 --minibatch_size 8 --num_workers 0
```

注意：如果正式实验降低了 batch 或 minibatch，后续报告里必须说明，因为这已经不是完全同一组实验设置。

## 14. 常见错误排查

### 1. `Food101N image not found`

检查：

```bash
head -5 data/Food-101N/meta/train.tsv
ls -l data/Food-101N/images/train | head
find data/Food-101N/images -type l | head
```

如果软链接断了，重新设置 `FOOD_IMAGES`，再跑 `prepare_food101n_metadata.py --link-images --overwrite`。

### 2. `cuda available: False`

检查：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

如果 `nvidia-smi` 正常但 PyTorch CUDA 不可用，通常是 PyTorch 安装成了 CPU 版，需要按官方 PyTorch selector 重装 GPU 版。

### 3. `ModuleNotFoundError`

在项目根目录补装依赖：

```bash
cd /home/hnh/mammoth_code_food101n
pip install -r requirements.txt
```

如果是可选包缺失，再按错误名单独安装。

### 4. `git clone` GitHub 报 `Recv failure: Connection reset by peer`

如果执行：

```bash
git clone --depth 1 https://github.com/wish44165/ntd.git external/ntd
```

出现：

```text
fatal: unable to access 'https://github.com/wish44165/ntd.git/':
Recv failure: Connection reset by peer
```

这不是 Python 虚拟环境错误，也不是 Food101N 代码错误，而是服务器访问 GitHub 时连接被重置。常见原因是服务器网络、代理、DNS、防火墙或 GitHub 访问不稳定。

先清理半截目录，再重试：

```bash
cd /home/hnh/mammoth_code_food101n
rm -rf external/ntd
git clone --depth 1 https://github.com/wish44165/ntd.git external/ntd
```

如果仍然失败，可以改用本地上传方式：

1. 在本地电脑打开：https://github.com/wish44165/ntd
2. 下载 `Code -> Download ZIP`。
3. 通过向日葵文件传输上传到服务器：

```text
/home/hnh/tmp/ntd-main.zip
```

4. 在服务器解压：

```bash
cd /home/hnh/mammoth_code_food101n
rm -rf external/ntd
mkdir -p external
python -m zipfile -e /home/hnh/tmp/ntd-main.zip /home/hnh/tmp/ntd_zip
mv /home/hnh/tmp/ntd_zip/ntd-main external/ntd
ls external/ntd/tasks/Food-101N | head
```

只要最后能看到 `external/ntd/tasks/Food-101N` 下的 JSON 文件，就可以继续 Food101N metadata 生成。

### 5. Kaggle CLI 报 `ProxyError: Cannot connect to proxy`

如果 `kaggle.json` 已经放好，但执行：

```bash
kaggle datasets list -s food-101n | head
```

出现：

```text
ProxyError('Cannot connect to proxy.', ConnectionResetError(104, 'Connection reset by peer'))
```

说明 Kaggle 账号凭证已通过文件检查，但服务器命令行访问 Kaggle 时走了一个不可用代理，或者当前网络把命令行连接重置了。这个错误不是 Food101N 代码问题，也不是 API token 内容错误。

先检查代理环境变量：

```bash
env | grep -i proxy
```

如果输出了 `http_proxy`、`https_proxy`、`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 等变量，可以先临时取消：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export KAGGLE_CONFIG_DIR=/home/hnh/.kaggle
kaggle datasets list -s food-101n | head
```

如果取消代理后仍然不通，说明服务器命令行直连 Kaggle 也不可用。此时不要继续卡在 Kaggle CLI，改用浏览器手动下载：

1. 在服务器浏览器打开 Kaggle Food-101N 数据集页面。
2. 点击 Download 下载 zip。
3. 在 WSL 中用 `find /mnt/c/Users -maxdepth 5 -iname '*food*101*n*.zip'` 找到下载文件。
4. 复制到 `/home/hnh/tmp/` 后解压。

示例：

```bash
find /mnt/c/Users -maxdepth 5 -iname '*food*101*n*.zip' 2>/dev/null
mkdir -p /home/hnh/tmp /home/hnh/datasets/Food-101N/raw
cp "/mnt/c/Users/a/Downloads/food-101n.zip" /home/hnh/tmp/food-101n.zip
python -m zipfile -e /home/hnh/tmp/food-101n.zip /home/hnh/datasets/Food-101N/raw
```

如果服务器浏览器能打开网页但下载很慢，也可以在本机下载 zip，然后通过向日葵文件传输上传到 `/home/hnh/tmp/`。

### 6. `Too many open files` 或 dataloader 卡住

先降低 workers：

```bash
--num_workers 0
```

确认能跑后再慢慢调到 2 或 4。

### 7. Food101N 不要设置 CIFAR 式噪声

Food101N 已经是真实 noisy label 数据集，正式命令应保持：

```bash
--noise_rate 0
```

不要加：

```bash
--noise_rate 0.2
--noise_rate 0.4
--noise_rate 0.6
--noise_type symm
--noise_type asym
```

这些是 CIFAR10/100 的合成噪声设置。

### 8. Food101N 正式训练显存接近 100% 或看起来不动

如果正式 OGC+SAP 日志停在类似：

```text
Task 1 - Epoch 2: 6% | 370/6560 [18:11<29:31:01, 17.17s/it]
```

同时 `nvidia-smi` 显示：

```text
GPU-Util 100%
Memory-Usage 24099MiB / 24564MiB
```

这通常不是卡死，而是模型仍在计算。Food101N 比 CIFAR10/100 重很多：

- Food101N 是 224x224 图片，CIFAR 是 32x32。
- Food101N 当前正式配置是 ResNet34。
- OGC+SAP 会同时用当前 batch、replay minibatch、buffer、SAP/OGC 相关中间量。
- `batch_size=32`、`minibatch_size=32`、`buffer_size=2000` 对 24GB GPU 会非常紧。
- PyTorch 会缓存/保留显存，所以看起来可能接近满显存。

先判断是否真卡住：

```bash
ps -p <PID> -o pid,etime,pcpu,pmem,cmd
tail -n 40 /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0.log
nvidia-smi
```

如果日志 iteration 数还在变化，说明仍在跑。如果每步已经到十几秒，完整 20 epoch 会非常慢，建议停止当前 `batch32/minibatch32` 任务，改用更稳的 `batch16/minibatch16` 先跑正式 seed0，并在报告中记录 batch 调整。

停止当前任务：

```bash
kill <PID>
sleep 5
nvidia-smi
```

如果进程没有退出，再用：

```bash
kill -9 <PID>
```

然后重跑较稳配置：

```bash
cd /home/hnh/mammoth_code_food101n
mkdir -p /home/hnh/food101n_runs/main
export PYTHONPATH=.
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FOOD101N_ROOT=/home/hnh/mammoth_code_food101n/data/Food-101N

COMMON_ARGS="--dataset seq-food101n \
  --food101n_root ${FOOD101N_ROOT} \
  --food101n_train_list meta/train.tsv \
  --food101n_test_list meta/test.tsv \
  --food101n_images_dir images \
  --food101n_classes_file meta/classes.txt \
  --backbone resnet34 \
  --n_epochs 20 \
  --batch_size 16 \
  --minibatch_size 16 \
  --lr 0.03 \
  --buffer_size 2000 \
  --num_workers 4 \
  --noise_rate 0"

nohup python -u main.py --model ogc-sap --enable_sap 1 \
  --sap_scale_coff 5000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 2000 \
  --savecheck last --seed 0 ${COMMON_ARGS} \
  > /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0_b16.log 2>&1 &
```

监控：

```bash
tail -f /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0_b16.log
nvidia-smi -l 2
```

## 15. 打包日志和结果

训练结束后，把日志和结果压缩到 `/home/hnh/tmp`：

```bash
cd /home/hnh
tar -czf /home/hnh/tmp/food101n_runs_$(date +%Y%m%d_%H%M%S).tar.gz \
  food101n_runs \
  mammoth_code_food101n/results \
  mammoth_code_food101n/checkpoints 2>/dev/null || true
```

然后通过向日葵文件传输把 `/home/hnh/tmp/food101n_runs_*.tar.gz` 下载回本地。

## 16. 最短执行清单

如果你只想按最短顺序执行，流程是：

```bash
cd /home/hnh
git clone git@github.com:nuohanhe69-pixel/noise-class-incremental-reproduction.git mammoth_code_food101n

source /home/hnh/software/miniconda3/etc/profile.d/conda.sh
conda activate /home/hnh/conda_envs/nrgp-mammoth

cd /home/hnh/mammoth_code_food101n
pip install -r requirements.txt

mkdir -p external
git clone --depth 1 https://github.com/wish44165/ntd.git external/ntd

# 上传并解压 Food101N 后，设置真实图片根目录
export FOOD_IMAGES=/home/hnh/datasets/Food-101N/raw/你的真实图片根目录

python scripts/prepare_food101n_metadata.py \
  --output-root data/Food-101N \
  --task-dir external/ntd/tasks/Food-101N \
  --task-rand 1 \
  --train-images-dir "$FOOD_IMAGES" \
  --test-images-dir "$FOOD_IMAGES" \
  --link-images \
  --overwrite

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

验证通过后，先跑 smoke，再跑正式训练。
