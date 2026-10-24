#!/usr/bin/env bash
# Run one persistent quota campaign. All attempts, including failures, are logged.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/env.sh"
cd "$root"
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-patchouli_lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-tactile-hardness-replenishment}"
: "${WANDB_API_KEY:?Forward WANDB_API_KEY as a SkyPilot secret}"
export WANDB_DISABLE_GIT=true
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
campaign_dir=/workspace/runs/convlstm-replenishment-20261022-v1
mkdir -p "$campaign_dir"
exec > >(tee -a "$campaign_dir/controller.log") 2>&1
python -m pip check
python scripts/patch_wandb_0179.py
python - <<'PY'
import os
import numpy as np
import tensorflow as tf
import wandb
x=np.load('/workspace/data/Own/X_4_raw.npy',mmap_mode='r',allow_pickle=False)
y=np.load('/workspace/data/Own/y_4_raw.npy',allow_pickle=True)
assert tuple(x.shape[1:])==(4,240,320,3) and len(x)==len(y),(x.shape,y.shape)
assert set(map(str,y.tolist()))=={'hardness'+str(i) for i in range(1,6)}
assert tf.config.list_physical_devices('GPU'),'No TensorFlow GPU detected'
with tf.device('/GPU:0'):
    check=tf.matmul(tf.ones((32,32)),tf.ones((32,32)))
assert 'GPU:0' in check.device
assert wandb.Api().viewer,'W&B authentication failed'
print('Replenishment preflight passed. Input shape:',x.shape,flush=True)
PY
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv,nounits --loop=2 >> "$campaign_dir/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
python -u scripts/replenish_convlstm.py \
    --workers "${REPLENISH_WORKERS:-1}" \
    --max-attempts "${REPLENISH_MAX_ATTEMPTS:-84}" \
    --output "$campaign_dir"
