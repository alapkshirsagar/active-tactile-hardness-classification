#!/usr/bin/env bash
# Source this file before invoking Python/Jupyter, or use scripts/run.sh.
_tactile_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CONDA_PREFIX="${_tactile_root}/.venv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# CUDA 11.2 must precede any host CUDA 12+ installation.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_HOME="${_tactile_root}/.cache/torch"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset _tactile_root
