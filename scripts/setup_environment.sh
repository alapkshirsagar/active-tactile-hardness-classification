#!/usr/bin/env bash
# Recreate the validated Linux x86_64 environment without changing system Python.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if [[ "$(uname -s)/$(uname -m)" != Linux/x86_64 ]]; then
    echo 'This lock targets Linux x86_64.' >&2
    exit 1
fi
mkdir -p .tools .cache/torch/hub/checkpoints
if [[ ! -x .tools/bin/micromamba ]]; then
    curl --fail --location --silent --show-error --retry 3 \
        https://micro.mamba.pm/api/micromamba/linux-64/2.3.3 \
        --output .tools/micromamba-2.3.3.tar.bz2
    tar -xjf .tools/micromamba-2.3.3.tar.bz2 -C .tools bin/micromamba
fi
printf '%s  %s\n' \
    9496f94a8b78c536573c93d946ec9bba74bd9ff79ee55aaa4b546e30db8f511b \
    .tools/bin/micromamba | sha256sum --check --status
export MAMBA_ROOT_PREFIX="$root/.tools/mamba"
if [[ ! -d .venv/conda-meta ]]; then
    if [[ -e .venv ]]; then
        echo '.venv exists but is not this conda environment; choose a clean checkout.' >&2
        exit 1
    fi
    .tools/bin/micromamba create --yes --prefix "$root/.venv" \
        --file environment/conda-linux-64.lock.txt
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 10, 11), sys.version'
# Build the imutils sdist with the same tooling as the validated installation.
.venv/bin/python -m pip install --disable-pip-version-check \
    setuptools==75.1.0 wheel==0.44.0
.venv/bin/python -m pip install --disable-pip-version-check --no-build-isolation \
    --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu \
    --requirement environment/requirements-linux-64.lock.txt

if [[ ! -f .cache/dinov2-e1277af.tar.gz ]]; then
    curl --fail --location --silent --show-error --retry 3 \
        https://codeload.github.com/facebookresearch/dinov2/tar.gz/e1277af2ba9496fbadf7aec6eba56e8d882d1e35 \
        --output .cache/dinov2-e1277af.tar.gz
fi
if [[ ! -f .cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth ]]; then
    curl --fail --location --silent --show-error --retry 3 \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
        --output .cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
fi
sha256sum --check environment/assets.sha256
# Existing models.py loads the default hub cache name. Populate it with a fixed
# source revision without changing the model or its preprocessing implementation.
mkdir -p .cache/torch/hub/facebookresearch_dinov2_main
tar -xzf .cache/dinov2-e1277af.tar.gz --strip-components=1 \
    -C .cache/torch/hub/facebookresearch_dinov2_main
printf '%s\n' e1277af2ba9496fbadf7aec6eba56e8d882d1e35 \
    > .cache/torch/hub/facebookresearch_dinov2_main/.tactile-revision
source scripts/env.sh
python -m ipykernel install --sys-prefix --name tactile-hardness \
    --display-name 'Tactile Hardness (Python 3.10 / TF 2.10)'
python -m pip check
echo 'Environment installed. Validate with: bash scripts/run.sh python scripts/verify_environment.py'
