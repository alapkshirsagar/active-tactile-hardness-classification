#!/usr/bin/env bash
# One full ConvLSTM variance trajectory; diagnostics do not count toward quotas.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/env.sh"
cd "$root"
export WANDB_MODE=online
export WANDB_PROJECT=tactile-hardness-smoke
export WANDB_ENTITY="${WANDB_ENTITY:-patchouli_lab}"
: "${WANDB_API_KEY:?Forward WANDB_API_KEY as a SkyPilot secret}"
export WANDB_TAGS=smoke-test,exclude-from-replenishment
smoke_dir="/workspace/runs/tactile-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$smoke_dir"
# The image creates an empty runs directory. Preserve existing contents if any.
if [[ -e "$root/runs" && ! -L "$root/runs" ]]; then
    rmdir "$root/runs" || { echo 'runs/ is not empty; refusing to replace it' >&2; exit 1; }
fi
ln -sfn "$smoke_dir" "$root/runs"
export SMOKE_DIAGNOSTICS_DIR="$smoke_dir"
exec > >(tee "$smoke_dir/smoke.log") 2>&1
printf 'Persistent diagnostics: %s\n' "$smoke_dir"
nvidia-smi
python -m pip check
python scripts/patch_wandb_0179.py
python - <<'PY'
import collections, importlib.metadata, json, os, platform
from pathlib import Path
import numpy as np
import tensorflow as tf
import wandb
out=Path(os.environ['SMOKE_DIAGNOSTICS_DIR'])
gpus=tf.config.list_physical_devices('GPU')
assert gpus, 'TensorFlow cannot see the GPU'
for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
with tf.device('/GPU:0'):
    result=tf.matmul(tf.ones((32,32)),tf.ones((32,32)))
assert 'GPU:0' in result.device and float(result[0,0])==32
base=Path('/workspace/data/Own')
x=np.load(base/'X_4_raw.npy',mmap_mode='r',allow_pickle=False)
y=np.load(base/'y_4_raw.npy',allow_pickle=True)
assert len(x)==len(y) and tuple(x.shape[1:])==(4,240,320,3), (x.shape,y.shape)
assert np.isfinite(x[0]).all(), 'First input sample contains non-finite values'
counts=collections.Counter(map(str,y.tolist()))
assert set(counts)=={'hardness'+str(i) for i in range(1,6)}, counts
assert min(counts.values())>=10, counts
viewer=wandb.Api().viewer
assert viewer, 'W&B authentication failed'
report={'python':platform.python_version(),
    'versions':{k:importlib.metadata.version(k) for k in ['tensorflow','keras','numpy','wandb','torch']},
    'gpu':[tf.config.experimental.get_device_details(g) for g in gpus],
    'data_shape':list(x.shape),'data_dtype':str(x.dtype),'class_counts':dict(counts),
    'files':{p.name:{'size_bytes':p.stat().st_size} for p in [base/'X_4_raw.npy',base/'y_4_raw.npy']},
    'wandb_authenticated':True,'wandb_project':os.environ['WANDB_PROJECT'],
    'image':'fheller0/tactile-hardness:tf210-smoke-v1',
    'wandb_compatibility_patch':'0.17.9 ServerSettings null handling',
    'purpose':'Environment/performance smoke test; excluded from replenishment quota'}
(out/'preflight.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report,indent=2,default=str),flush=True)
PY
# Existing launcher records GPU utilization every 2 s, resource usage and timing.
# W&B run.finish() synchronizes online metrics before the training process exits.
bash scripts/benchmark_ablation.sh ConvLSTM hardness
printf 'Smoke run completed. Diagnostics: %s\n' "$smoke_dir"
