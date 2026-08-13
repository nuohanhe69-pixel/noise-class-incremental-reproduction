#!/usr/bin/env bash

set -u
set -o pipefail

SAP_RUN_MODE="${1:-}"
if [ "${SAP_RUN_MODE}" != "smoke" ] && [ "${SAP_RUN_MODE}" != "full" ]; then
  echo "Usage: bash scripts/run_cifar100_dgc_sap_server.sh {smoke|full}" >&2
  exit 2
fi

SAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAP_PROJECT_DIR="$(cd "${SAP_SCRIPT_DIR}/.." && pwd)"
SAP_EXPECTED_PROJECT_DIR="${SAP_EXPECTED_PROJECT_DIR:-/home/hnh/mammoth_cifar100/mammoth_code}"
SAP_CONDA_SH="${SAP_CONDA_SH:-/home/hnh/software/miniconda3/etc/profile.d/conda.sh}"
SAP_CONDA_ENV="${SAP_CONDA_ENV:-/home/hnh/conda_envs/nrgp-mammoth}"
SAP_GPU="${SAP_GPU:-0}"
SAP_SEED="${SAP_SEED:-0}"
SAP_PATCHES="${SAP_PATCHES:-20000}"
SAP_RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SAP_RUN_NAME="cifar100_symm40_dgc_sap_${SAP_RUN_MODE}_seed${SAP_SEED}_${SAP_RUN_TAG}"
SAP_RUN_ROOT="/home/hnh/dgc_sap_runs/${SAP_RUN_NAME}"
SAP_LOG="${SAP_RUN_ROOT}/train.log"
SAP_MANIFEST="${SAP_RUN_ROOT}/manifest.txt"
SAP_RESULTS="${SAP_RUN_ROOT}/results"
SAP_CHECKPOINTS="${SAP_RUN_ROOT}/checkpoints"

if [ "${SAP_PROJECT_DIR}" != "${SAP_EXPECTED_PROJECT_DIR}" ]; then
  echo "ERROR: expected project at ${SAP_EXPECTED_PROJECT_DIR}, found ${SAP_PROJECT_DIR}" >&2
  exit 2
fi
if [ ! -f "${SAP_CONDA_SH}" ]; then
  echo "ERROR: conda init script not found: ${SAP_CONDA_SH}" >&2
  exit 2
fi
if [ ! -d "${SAP_CONDA_ENV}" ]; then
  echo "ERROR: conda environment not found: ${SAP_CONDA_ENV}" >&2
  exit 2
fi

mkdir -p "${SAP_RUN_ROOT}" "${SAP_RESULTS}" "${SAP_CHECKPOINTS}"
source "${SAP_CONDA_SH}"
conda activate "${SAP_CONDA_ENV}"
cd "${SAP_PROJECT_DIR}" || exit 2

if [ "${SAP_RUN_MODE}" = "smoke" ]; then
  SAP_STOP_AFTER=3
  SAP_SAVE_MODE=task
else
  SAP_STOP_AFTER=10
  SAP_SAVE_MODE=last
fi

SAP_COMMAND=(
  python -u main.py
  --dataset seq-cifar100
  --model dgc-sap
  --backbone resnet18
  --noise_type symm
  --noise_rate 0.4
  --n_epochs 50
  --batch_size 32
  --minibatch_size 32
  --buffer_size 2000
  --optimizer sgd
  --lr 0.03
  --optim_mom 0
  --optim_wd 0
  --optim_nesterov 0
  --use_aer 1
  --sample_selection_strategy abs
  --alpha_sample_insertion 0.75
  --buffer_fitting_epochs 0
  --ogc_loss_weight 0.3
  --ogc_low_conf_weight 0.4
  --ogc_buffer_penalty_coeff 1.8
  --sap_reference_per_class 15
  --sap_scale 3000
  --sap_max_activation_patches "${SAP_PATCHES}"
  --sap_batch_size 32
  --sap_dry_run 0
  --seed "${SAP_SEED}"
  --num_workers 4
  --stop_after "${SAP_STOP_AFTER}"
  --savecheck "${SAP_SAVE_MODE}"
  --save_checkpoint_mode safe
  --ckpt_name "${SAP_RUN_NAME}"
  --results_path "${SAP_RESULTS}"
  --checkpoint_path "${SAP_CHECKPOINTS}"
)

{
  echo "RUN_NAME=${SAP_RUN_NAME}"
  echo "RUN_MODE=${SAP_RUN_MODE}"
  echo "START_TIME=$(date '+%F %T %Z')"
  echo "PROJECT_DIR=${SAP_PROJECT_DIR}"
  echo "RUN_ROOT=${SAP_RUN_ROOT}"
  echo "CONDA_ENV=${SAP_CONDA_ENV}"
  echo "CUDA_VISIBLE_DEVICES=${SAP_GPU}"
  echo "SEED=${SAP_SEED}"
  echo "PATCHES=${SAP_PATCHES}"
  echo "STOP_AFTER=${SAP_STOP_AFTER}"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "GIT_BRANCH=$(git branch --show-current)"
  echo "GIT_DIRTY_BEGIN"
  git status --short
  echo "GIT_DIRTY_END"
  which python
  python --version
  python -c 'import torch, sklearn; assert torch.cuda.is_available(), "CUDA is required"; print("torch=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("cuda=" + str(torch.version.cuda)); print("gpu=" + torch.cuda.get_device_name(0)); print("gpu_memory_bytes=" + str(torch.cuda.get_device_properties(0).total_memory)); print("sklearn=" + sklearn.__version__)'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  printf 'COMMAND='
  printf '%q ' "CUDA_VISIBLE_DEVICES=${SAP_GPU}" "${SAP_COMMAND[@]}"
  printf '\n'
} | tee "${SAP_MANIFEST}" "${SAP_LOG}"

CUDA_VISIBLE_DEVICES="${SAP_GPU}" "${SAP_COMMAND[@]}" 2>&1 | tee -a "${SAP_LOG}"
SAP_STATUS=${PIPESTATUS[0]}

{
  echo "TRAIN_STATUS=${SAP_STATUS}"
  echo "END_TIME=$(date '+%F %T %Z')"
  echo "CHECKPOINT_FILES=$(find "${SAP_CHECKPOINTS}" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
} | tee -a "${SAP_MANIFEST}" "${SAP_LOG}"

if [ "${SAP_STATUS}" -ne 0 ]; then
  echo "ERROR: run failed; inspect ${SAP_LOG}" >&2
  exit "${SAP_STATUS}"
fi

python - "${SAP_CHECKPOINTS}" "${SAP_STOP_AFTER}" <<'PY' | tee -a "${SAP_LOG}"
import sys
from pathlib import Path
import torch

checkpoint_dir = Path(sys.argv[1])
expected_tasks = int(sys.argv[2])
checkpoint_paths = sorted(checkpoint_dir.glob('*.pt'), key=lambda path: path.stat().st_mtime)
if not checkpoint_paths:
    raise SystemExit('no checkpoint was produced')
saved = torch.load(checkpoint_paths[-1], map_location='cpu', weights_only=False)
sap_state = saved.get('sap_state')
if not sap_state:
    raise SystemExit('latest checkpoint has no sap_state')
history = sap_state.get('history', [])
if len(history) != expected_tasks:
    raise SystemExit(f'unexpected SAP history length: {len(history)} != {expected_tasks}')
allowed = {
    'SAP_EXECUTED',
    'SAP_SKIPPED_INSUFFICIENT_REFERENCE',
    'SAP_SKIPPED_EXCESSIVE_FALLBACK',
}
statuses = [event.get('status') for event in history]
if any(status not in allowed for status in statuses):
    raise SystemExit(f'unexpected SAP status: {statuses}')
for event in history:
    if event.get('status') == 'SAP_EXECUTED':
        if len(event.get('layer_stats', {})) != 10:
            raise SystemExit('executed SAP event does not contain 10 layer stats')
        if event.get('max_non_target_state_delta') != 0:
            raise SystemExit('non-target model state changed during SAP')
        if set(event.get('comparisons', {})) != {'reference', 'replay_buffer'}:
            raise SystemExit('executed SAP event is missing reference/replay diagnostics')
print('LATEST_CHECKPOINT=' + str(checkpoint_paths[-1]))
print('SAP_STATUSES=' + ','.join(statuses))
print('SAP_REFERENCE_COUNT=' + str(sum(len(items) for items in sap_state['reference_memory']['tasks'].values())))
print('SERVER_RUN_VALIDATION_OK')
PY
SAP_VALIDATE_STATUS=${PIPESTATUS[0]}
if [ "${SAP_VALIDATE_STATUS}" -ne 0 ]; then
  echo "ERROR: checkpoint validation failed; inspect ${SAP_LOG}" >&2
  exit "${SAP_VALIDATE_STATUS}"
fi

echo "ALL_DONE RUN_ROOT=${SAP_RUN_ROOT}" | tee -a "${SAP_MANIFEST}" "${SAP_LOG}"
