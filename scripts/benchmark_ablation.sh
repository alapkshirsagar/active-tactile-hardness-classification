#!/usr/bin/env bash
# One complete, historical-protocol dropout ablation for runtime measurement.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/env.sh"
model="${1:-DinoNN}"
dataset="${2:-yuan}"
case "$model/$dataset" in
    DinoNN/yuan) description='DINOv2NN / Yuan FLAT / comparison class 27'; initial_epochs=50; iteration_epochs=25 ;;
    ConvLSTM/hardness) description='ConvLSTM / own data / comparison class hardness3'; initial_epochs=100; iteration_epochs=50 ;;
    *) echo 'Usage: benchmark_ablation.sh [DinoNN yuan | ConvLSTM hardness]' >&2; exit 2 ;;
esac
run_label="timing_${dataset}_${model,,}_d020_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$root/runs/$run_label"
mkdir -p "$run_dir" "$root/hardness_classification/AdaptiveModels"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$run_dir"
export WANDB_DISABLE_GIT=true
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1
cd "$root/hardness_classification"
printf 'Run directory: %s\n' "$run_dir"
printf '%s / dropout 0.2 / variance\n' "$description"
printf '%s initial epochs + 5 iterations of %s epochs = %s epochs.\n' \
    "$initial_epochs" "$iteration_epochs" "$((initial_epochs + 5 * iteration_epochs))"
printf 'Existing acquisition and uncertainty behavior preserved for timing.\n'
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv,nounits --loop=2 > "$run_dir/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_dir/started_at.txt"
started=$(date +%s)
set +e
python "$root/scripts/time_command.py" "$run_dir/resource_usage.json" \
    python -u ./leave_one_out.py \
    --comp_index 2 --strat variance --prefix "$run_label" \
    --model "$model" --dataset "$dataset" --dropout 0.2 \
    2>&1 | tee "$run_dir/train.log"
status=${PIPESTATUS[0]}
set -e
finished=$(date +%s)
elapsed=$((finished - started))
printf '{"exit_code": %d, "elapsed_seconds": %d}\n' "$status" "$elapsed" \
    > "$run_dir/timing.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_dir/finished_at.txt"
printf '\nFinished: exit=%s, elapsed=%ss. Results: %s\n' "$status" "$elapsed" "$run_dir"
exit "$status"
