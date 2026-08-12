# 通过向日葵在远程服务器运行 CIFAR-10 ER-ACE 四线 Loss 实验

## 1. 已确认的服务器信息

以 Codex 会话 `019f92e8-4e9e-7c43-9c79-d158dac71e5d` 中的已有运行记录为准：

| 项目 | 值 |
|---|---|
| 远程形态 | Windows 主机 + WSL Linux，通过向日葵远程 |
| WSL 用户目录 | `/home/hnh` |
| 已有 CIFAR-10 项目 | `/home/hnh/mammoth_code_cifar10` |
| Conda 初始化 | `/home/hnh/software/miniconda3/etc/profile.d/conda.sh` |
| Conda 环境 | `/home/hnh/conda_envs/nrgp-mammoth` |
| Python 期望路径 | `/home/hnh/conda_envs/nrgp-mammoth/bin/python` |
| GPU | 单张 NVIDIA RTX 4090 |
| 远程 Windows 桌面 | `/mnt/c/Users/a/Desktop` |
| 后台运行 | `screen` |

本实验是 `Seq-CIFAR10 / ER-ACE / 40% symmetric noise / Task 2 epochs [5,10)` 的四线图。不涉及 DGC，不应上传或覆盖服务器上手工修改过的模型文件。

## 2. 上传包内容

上传包只包含纯 ER-ACE 四线实验必须文件：

```text
datasets/utils/continual_dataset.py
models/er_ace.py
utils/buffer.py
utils/loss_trace.py
scripts/plot_loss_trace.py
scripts/run_cifar10_erace_four_loss_server.sh
tests/test_loss_trace.py
docs/loss_trace_four_curves.md
docs/loss_trace_server_sunlogin_runbook.md
```

服务器上直接使用已验证能读取数据的 CIFAR-10 项目 `/home/hnh/mammoth_code_cifar10`。覆盖前逐文件备份；不从 CIFAR-100 项目复制代码，也不重新上传数据集。

## 3. 在本机通过向日葵上传

### 3.1 确认要上传的两个文件

本机 Mac 上的文件是：

```text
/Users/hunk/Documents/噪声类增量项目复现/server_upload/loss_trace_cifar10_upload_20260801.tar.gz
/Users/hunk/Documents/噪声类增量项目复现/server_upload/loss_trace_cifar10_upload_20260801.tar.gz.sha256
```

根据服务器截图和用户确认，远程已有 CIFAR-10 数据与相同 Conda 环境，因此不需要上传数据包。

### 3.2 打开向日葵文件传输

1. 连接到远程 Windows 主机。
2. 打开向日葵的“文件传输”。
3. 本地窗格进入上述 `server_upload` 目录。
4. 远程窗格进入 `C:\Users\a\Desktop`。
5. 只把代码包 `.tar.gz` 和 `.sha256` 两个文件上传到远程桌面。
6. 等待两个文件都显示传输完成，不要在传输未完成时解压。

## 4. 在远程 WSL 校验上传文件

打开远程 Windows 中的 WSL 终端，执行：

```bash
ls -lh /mnt/c/Users/a/Desktop/loss_trace_cifar10_upload_20260801.tar.gz
ls -lh /mnt/c/Users/a/Desktop/loss_trace_cifar10_upload_20260801.tar.gz.sha256
```

如果 `/mnt/c/Users/a/Desktop` 不存在，先执行：

```bash
ls /mnt/c/Users
```

把后续命令中的 `a` 替换成实际 Windows 用户名。

复制到 WSL 临时目录：

```bash
mkdir -p /home/hnh/tmp/loss_trace_cifar10_upload_20260801
cp -v /mnt/c/Users/a/Desktop/loss_trace_cifar10_upload_20260801.tar.gz /home/hnh/tmp/loss_trace_cifar10_upload_20260801/
cp -v /mnt/c/Users/a/Desktop/loss_trace_cifar10_upload_20260801.tar.gz.sha256 /home/hnh/tmp/loss_trace_cifar10_upload_20260801/
cd /home/hnh/tmp/loss_trace_cifar10_upload_20260801
sha256sum -c loss_trace_cifar10_upload_20260801.tar.gz.sha256
```

正常必须输出：

```text
loss_trace_cifar10_upload_20260801.tar.gz: OK
```

再只读检查压缩包内容：

```bash
tar -tzf loss_trace_cifar10_upload_20260801.tar.gz
```

文件列表必须与第 2 节一致。

## 5. 检查服务器资源，不要盲目叠加训练

```bash
df -h /home/hnh
nvidia-smi
screen -ls
ps -efww | grep '[m]ain.py'
```

如果 GPU 上有必须保留的实验，不要执行本次正式训练，等它结束。不要使用无区分的 `pkill -f main.py`，它可能终止其他人或其他实验。

## 6. 确认 CIFAR-10 项目并备份将被覆盖的文件

设置实验项目路径：

```bash
SERVER_EXP=/home/hnh/mammoth_code_cifar10
```

以下第 6–10 节应在同一个 WSL 终端中连续执行。如果向日葵断线或重新打开终端，需要重新执行上面的赋值。

确认是 CIFAR-10 项目，不要在 `/home/hnh/mammoth_cifar100/mammoth_code` 中操作：

```bash
cd "${SERVER_EXP}"
pwd
test -f main.py && echo PROJECT_OK
test -f datasets/seq_cifar10.py && echo CIFAR10_CODE_OK
find data /home/hnh/datasets -maxdepth 4 -type f -name data_batch_1 -print 2>/dev/null | head
```

只要已有 CIFAR-10 项目之前能正常运行，不需要改 `base_path`、创建数据软链接或搬移数据。上面 `find` 如果没找到文件，也先不修改配置，由第 10 节冒烟测试验证实际读取路径。

建立带时间戳的逐文件备份：

```bash
LOSS_TRACE_BACKUP="/home/hnh/backup_models/loss_trace_cifar10_before_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOSS_TRACE_BACKUP}"
cd "${SERVER_EXP}"
for LOSS_TRACE_FILE in \
  datasets/utils/continual_dataset.py \
  models/er_ace.py \
  utils/buffer.py \
  utils/loss_trace.py \
  scripts/plot_loss_trace.py \
  scripts/run_cifar10_erace_four_loss_server.sh \
  tests/test_loss_trace.py \
  docs/loss_trace_four_curves.md \
  docs/loss_trace_server_sunlogin_runbook.md
do
  if [ -e "${LOSS_TRACE_FILE}" ]; then
    mkdir -p "${LOSS_TRACE_BACKUP}/$(dirname "${LOSS_TRACE_FILE}")"
    cp -a "${LOSS_TRACE_FILE}" "${LOSS_TRACE_BACKUP}/${LOSS_TRACE_FILE}"
  fi
done
echo "LOSS_TRACE_BACKUP=${LOSS_TRACE_BACKUP}"
find "${LOSS_TRACE_BACKUP}" -type f -print
```

保留屏幕中输出的 `LOSS_TRACE_BACKUP=...`，它是需要回滚时的备份目录。这个补丁不包含 `models/er_ace_aer_abs.py` 或 `models/dgc.py`，所以不会覆盖服务器上的当前模型实现。

## 7. 在现有 CIFAR-10 项目中安装上传补丁

```bash
cd /home/hnh/tmp/loss_trace_cifar10_upload_20260801
tar -xzf loss_trace_cifar10_upload_20260801.tar.gz -C "${SERVER_EXP}"
cd "${SERVER_EXP}"
chmod +x scripts/run_cifar10_erace_four_loss_server.sh scripts/plot_loss_trace.py
```

确认关键文件：

```bash
grep -n "enable_loss_trace\|loss_trace_hard_ratio" models/er_ace.py utils/loss_trace.py
grep -n "sample_ids\|source_task_ids" datasets/utils/continual_dataset.py utils/buffer.py
grep -n "n_epochs 50\|loss_trace_start_epoch 5\|loss_trace_end_epoch 9" scripts/run_cifar10_erace_four_loss_server.sh
```

## 8. 激活环境并做依赖检查

```bash
source /home/hnh/software/miniconda3/etc/profile.d/conda.sh
conda activate /home/hnh/conda_envs/nrgp-mammoth
cd "${SERVER_EXP}"
which python
python --version
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
```

`which python` 应指向：

```text
/home/hnh/conda_envs/nrgp-mammoth/bin/python
```

GPU 必须是可用状态。

检查作图依赖：

```bash
python -c 'import pandas, matplotlib; print("plot dependencies OK")'
```

如果缺少，在当前 Conda 环境安装：

```bash
python -m pip install pandas matplotlib
```

## 9. 静态检查和单元测试

```bash
cd "${SERVER_EXP}"
python -m py_compile \
  utils/loss_trace.py \
  models/er_ace.py \
  datasets/utils/continual_dataset.py \
  utils/buffer.py \
  scripts/plot_loss_trace.py
python -m unittest tests.test_loss_trace -v
bash -n scripts/run_cifar10_erace_four_loss_server.sh
```

正常结果：

```text
Ran 3 tests
OK
```

检查命令行参数；Mammoth 的 `--help` 也需要同时给 `buffer_size`：

```bash
python main.py \
  --model er-ace \
  --dataset seq-cifar10 \
  --buffer_size 500 \
  --help | grep -n "loss_trace\|use_custom_classifier\|minibatch_size"
```

## 10. 先跑两个 task 的极小冒烟实验

正式训练前必须先跑冒烟测试：

```bash
cd "${SERVER_EXP}"
mkdir -p output/loss_trace_smoke run_logs
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset seq-cifar10 \
  --model er-ace \
  --backbone resnet18 \
  --noise_type symm \
  --noise_rate 0.4 \
  --n_epochs 1 \
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
  --num_workers 0 \
  --debug_mode 1 \
  --stop_after 2 \
  --enable_loss_trace 1 \
  --loss_trace_task 1 \
  --loss_trace_start_epoch 0 \
  --loss_trace_end_epoch 0 \
  --loss_trace_hard_ratio 0.2 \
  --loss_trace_dir output/loss_trace_smoke \
  2>&1 | tee run_logs/cifar10_erace_four_loss_smoke.log
```

确认退出码：

```bash
echo ${PIPESTATUS[0]}
```

必须是 `0`。

检查输出：

```bash
find output/loss_trace_smoke -maxdepth 2 -type f -print
find output/loss_trace_smoke -name iteration_curves.csv -exec wc -l {} \;
find output/loss_trace_smoke -name sample_losses.csv -exec wc -l {} \;
```

必须同时存在：

```text
metadata.json
iteration_curves.csv
sample_losses.csv
```

且两个 CSV 不能只有表头。

## 11. 启动正式实验

再次确认 GPU 没有需要保留的训练：

```bash
nvidia-smi
ps -efww | grep '[m]ain.py'
```

确认没有同名 screen：

```bash
screen -ls
```

后台启动 seed 0：

```bash
screen -dmS cifar10_erace_four_loss_seed0 bash -lc '
cd /home/hnh/mammoth_code_cifar10
TRACE_SEED=0 CUDA_VISIBLE_DEVICES=0 bash scripts/run_cifar10_erace_four_loss_server.sh
'
```

确认 screen：

```bash
screen -ls
```

应看到：

```text
cifar10_erace_four_loss_seed0    Detached
```

确认训练进程：

```bash
ps -efww | grep '[m]ain.py' | grep 'seq-cifar10' | grep 'er-ace'
nvidia-smi
```

## 12. 运行期间查看进度

列出当前日志：

```bash
cd /home/hnh/mammoth_code_cifar10
ls -lt run_logs/cifar10_erace_four_loss_seed0_*.log | head
```

实时查看：

```bash
tail -f "$(ls -t run_logs/cifar10_erace_four_loss_seed0_*.log | head -1)"
```

退出 `tail -f` 按 `Ctrl+C`，不会停止后台训练。

也可进入 screen：

```bash
screen -r cifar10_erace_four_loss_seed0
```

从 screen 安全退回后台：

1. 按 `Ctrl+A`。
2. 松开。
3. 按 `D`。

不要在 screen 中按 `Ctrl+C`，否则会中断训练。

## 13. 判定是完成还是失败

实验结束后，同名 screen 自动消失是正常的。不能只根据 screen 消失判断成功，必须检查日志：

```bash
cd /home/hnh/mammoth_code_cifar10
LATEST_TRACE_LOG="$(ls -t run_logs/cifar10_erace_four_loss_seed0_*.log | head -1)"
echo "${LATEST_TRACE_LOG}"
tail -n 60 "${LATEST_TRACE_LOG}"
grep -n "TRAIN_STATUS=0\|ALL_DONE\|Traceback\|ERROR:" "${LATEST_TRACE_LOG}"
```

成功时必须同时有：

```text
TRAIN_STATUS=0
ALL_DONE
```

并且不能有 Python traceback。

## 14. 校验四线数据与图片

从日志取出真实 trace 目录：

```bash
TRACE_RUN_DIR="$(grep '^TRACE_RUN_DIR=' "${LATEST_TRACE_LOG}" | tail -1 | cut -d= -f2-)"
echo "${TRACE_RUN_DIR}"
ls -lh "${TRACE_RUN_DIR}"
```

必须看到：

```text
metadata.json
sample_losses.csv
iteration_curves.csv
four_loss_curves.png
four_loss_curves.pdf
```

检查元数据：

```bash
python -m json.tool "${TRACE_RUN_DIR}/metadata.json"
```

重点确认：

```text
dataset = seq-cifar10
model = er_ace
seed = 0
noise_type = symmetric
noise_rate = 0.4
hard_old_ratio = 0.2
loss_timing = before optimizer.step
```

检查曲线表头和前 5 行：

```bash
sed -n '1,6p' "${TRACE_RUN_DIR}/iteration_curves.csv"
```

检查样本表头：

```bash
sed -n '1,3p' "${TRACE_RUN_DIR}/sample_losses.csv"
```

日志中还必须有类似：

```text
X_EPOCH_RANGE=5.000000..9.99...
NON_EMPTY_CURVE_POINTS={'noisy_loss': ..., 'hard_old_loss': ..., 'easy_old_loss': ..., 'new_loss': ...}
```

四个值都应该大于 0。

## 15. 把实验结果放到远程 Windows 桌面

运行脚本会自动创建结果压缩包和 SHA-256 文件。查看：

```bash
cd /home/hnh/mammoth_code_cifar10
LATEST_RESULT_ARCHIVE="$(ls -t artifacts/cifar10_erace_four_loss_seed0_*.tar.gz | head -1)"
echo "${LATEST_RESULT_ARCHIVE}"
ls -lh "${LATEST_RESULT_ARCHIVE}" "${LATEST_RESULT_ARCHIVE}.sha256"
```

复制到 Windows 桌面：

```bash
cp -v "${LATEST_RESULT_ARCHIVE}" /mnt/c/Users/a/Desktop/
cp -v "${LATEST_RESULT_ARCHIVE}.sha256" /mnt/c/Users/a/Desktop/
```

远程 Windows 桌面应出现两个文件。

## 16. 通过向日葵把结果下载回本机

1. 打开向日葵“文件传输”。
2. 远程窗格进入 `C:\Users\a\Desktop`。
3. 选中最新的 `cifar10_erace_four_loss_seed0_*.tar.gz` 及对应 `.sha256`。
4. 本地窗格选择结果保存目录。
5. 下载两个文件。
6. 本机 Mac 终端进入下载目录，执行：

```bash
shasum -a 256 -c cifar10_erace_four_loss_seed0_*.tar.gz.sha256
```

必须显示 `OK`，然后解压：

```bash
tar -xzf cifar10_erace_four_loss_seed0_*.tar.gz
```

解压后包含：

```text
metadata.json
sample_losses.csv
iteration_curves.csv
four_loss_curves.png
four_loss_curves.pdf
cifar10_erace_four_loss_seed0_*.log
```

## 17. 可选的多 seed 实验

图 1 本身的具体 seed 未在论文中公开。建议先用 seed 0 完成单次过程图。如需要统计置信带，可以再运行一组预先声明的 5 seeds，但这些 seed 是本项目复现协议，不能写成论文原始 seed。

例如下一次运行 seed 1：

```bash
screen -dmS cifar10_erace_four_loss_seed1 bash -lc '
cd /home/hnh/mammoth_code_cifar10
TRACE_SEED=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_cifar10_erace_four_loss_server.sh
'
```

单张 RTX 4090 上建议不同 seed 串行运行，以减少显存竞争和非确定性差异。

## 18. 常见问题

### 18.1 `Buffer size not found`

Mammoth 加载 `--help` 也需要 `--buffer_size 500`，使用第 9 节的完整检查命令。

### 18.2 `screen` 很快消失

可能是脚本完成，也可能是启动失败。必须查看最新 `run_logs/cifar10_erace_four_loss_seed*.log`，不能只看 `screen -ls`。

### 18.3 日志出现 `No module named 'models.'`

该仓库某些配置加载路径会先打印 warning 再回退。如果后面已打印完整 configuration 并进入 `Task 1 - Epoch 1`，该 warning 本身不等于训练失败；如果出现 traceback 或没有进入训练，则按失败处理。

### 18.4 没有图片

检查日志是否有：

```text
PLOT_SKIPPED=missing pandas or matplotlib
```

如有，安装依赖后手动画图：

```bash
python scripts/plot_loss_trace.py "${TRACE_RUN_DIR}" --smooth-window 5
```

### 18.5 CUDA OOM

本实验只启动一个 batch 32 + replay batch 32 的 ResNet18 进程，RTX 4090 正常应足够。如果 OOM，先检查 `nvidia-smi` 是否有其他训练；不要立即改 batch size，否则会改变实验协议。

### 18.6 CSV 存在但四条线有空组

单个 iteration 某组可能没有样本，该点为 `NaN` 是正常的。需要检查的是整个 epochs 5–9 区间内每条线都应该有非空点，运行脚本已自动完成这项检查。

### 18.7 补丁后静态检查或冒烟测试失败

先保留失败日志，不要立即启动正式实验。如果确定需要恢复补丁前的文件，使用第 6 节实际输出的备份路径：

```bash
SERVER_EXP=/home/hnh/mammoth_code_cifar10
LOSS_TRACE_BACKUP=/home/hnh/backup_models/loss_trace_cifar10_before_YYYYMMDD_HHMMSS
test -d "${LOSS_TRACE_BACKUP}" || { echo "STOP: backup not found"; exit 1; }
cd "${LOSS_TRACE_BACKUP}"
find . -type f -print
cp -a . "${SERVER_EXP}/"
```

把 `YYYYMMDD_HHMMSS` 换成第 6 节记录的真实时间戳。该命令只恢复曾经存在并被备份的文件；补丁新增的记录和作图文件可以保留，因为 `--enable_loss_trace` 默认关闭，不会改变原有训练。
