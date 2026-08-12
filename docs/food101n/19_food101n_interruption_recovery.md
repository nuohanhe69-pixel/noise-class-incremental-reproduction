# Food101N 中断恢复与重跑方案

本文记录服务器上 Food101N 实验跑到一半中断后的排查、恢复和推荐重跑方案。

## 当前现象判断

截图中日志停在：

```text
Task 3 - Epoch 14: 67% ... ^C
```

随后 `nvidia-smi` 显示显存约 `566MiB / 24564MiB`，GPU 利用率 `0%`。这说明训练进程已经不在运行。`^C` 可能只是停止了 `tail -f`，但因为 GPU 上已经没有 Python 训练进程，所以实际训练也已经停止或退出。

## 第一步：确认进程和日志

在服务器终端执行：

```bash
cd /home/hnh/mammoth_code_food101n

source /home/hnh/software/miniconda3/etc/profile.d/conda.sh
conda activate /home/hnh/conda_envs/nrgp-mammoth

ps -u "$USER" -f | grep '[m]ain.py'
nvidia-smi

LOG=/home/hnh/food101n_runs/main/food101n_ogc_sap_seed0_b16.log
tail -120 "$LOG"
grep -nE "Traceback|Error|Killed|CUDA out of memory|SIGINT|SIGTERM|KeyboardInterrupt|Checkpoint|completed|Logging results" "$LOG" | tail -80
```

如果 `ps` 没有输出训练进程，并且 `nvidia-smi` 只有几百 MB 显存占用，就说明这次训练已经结束或被中断了。

## 第二步：检查有没有 checkpoint

执行：

```bash
cd /home/hnh/mammoth_code_food101n

find /home/hnh/mammoth_code_food101n/checkpoints \
  -type f -name "*.pt" \
  -printf "%TY-%Tm-%Td %TH:%TM %p\n" 2>/dev/null | sort | tail -30

find /home/hnh/mammoth_code_food101n/checkpoints/paused \
  -type f -name "*.pt" \
  -printf "%TY-%Tm-%Td %TH:%TM %p\n" 2>/dev/null | sort | tail -20
```

解释：

- `--savecheck last`：只在最后一个 task 结束后保存正式 checkpoint。当前中断在 Task 3，通常没有可用于正式续跑的 checkpoint。
- `checkpoints/paused/*.pt`：如果程序捕获到 `SIGINT` 或 `SIGTERM`，可能会临时保存中断 checkpoint。但它一般是 task 中途状态，不适合作为论文复现实验的正式结果。
- `--savecheck task`：每个 task 结束保存一次，适合 Food101N 这种长实验。

## 推荐方案：从头重跑，并改成每个 task 保存

当前这次如果没有 task checkpoint，最稳妥的方案是从头重跑。新命令把 `--savecheck last` 改成 `--savecheck task`，这样以后即使中断，也能从上一个完整 task 后继续。

```bash
cd /home/hnh/mammoth_code_food101n

source /home/hnh/software/miniconda3/etc/profile.d/conda.sh
conda activate /home/hnh/conda_envs/nrgp-mammoth

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

nohup python -u main.py \
  --model ogc-sap \
  --enable_sap 1 \
  --sap_scale_coff 5000 \
  --ogc_loss_weight 0.3 \
  --ogc_low_conf_weight 0.3 \
  --ogc_buffer_penalty_coeff 2.0 \
  --sap_retain_samples 2000 \
  --savecheck task \
  --ckpt_name food101n_ogc_sap_seed0_b16_taskckpt \
  --seed 0 \
  ${COMMON_ARGS} \
  > /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0_b16_taskckpt.log 2>&1 &
```

查看运行状态：

```bash
tail -f /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0_b16_taskckpt.log
```

另开一个终端查看 GPU：

```bash
nvidia-smi -l 2
```

查看 task checkpoint：

```bash
find /home/hnh/mammoth_code_food101n/checkpoints \
  -type f -name "food101n_ogc_sap_seed0_b16_taskckpt*.pt" \
  -printf "%TY-%Tm-%Td %TH:%TM %p\n" | sort | tail -20
```

## 以后如果从 task checkpoint 续跑

如果以后已经使用 `--savecheck task`，并且看到类似：

```text
checkpoints/food101n_ogc_sap_seed0_b16_taskckpt_2.pt
```

这通常表示 Task 3 结束后的 checkpoint，因为任务编号从 0 开始：`_0` 对应 Task 1 后，`_1` 对应 Task 2 后，`_2` 对应 Task 3 后。

从 Task 4 继续时使用：

```bash
CKPT=/home/hnh/mammoth_code_food101n/checkpoints/food101n_ogc_sap_seed0_b16_taskckpt_2.pt

nohup python -u main.py \
  --loadcheck "$CKPT" \
  --start_from 3 \
  --savecheck task \
  --ckpt_name food101n_ogc_sap_seed0_b16_resume_from_task4 \
  ${COMMON_ARGS} \
  > /home/hnh/food101n_runs/main/food101n_ogc_sap_seed0_b16_resume_from_task4.log 2>&1 &
```

注意：不要从 task 中途的 `paused` checkpoint 当作正式复现实验结果。正式结果最好从完整 task checkpoint 续跑，或者从头重跑。

## 是否需要 sudo

不需要。实验、数据、环境、日志都在 `/home/hnh` 下，普通用户权限即可。不要用 `sudo python main.py`，否则可能导致 conda 环境、文件权限和日志权限混乱。
