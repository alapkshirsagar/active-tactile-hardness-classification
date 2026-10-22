# Docker and SkyPilot on RunPod

The image recreates the tested micromamba/pip environment using the existing
native and hashed Python locks. This checkout has no `uv.lock` or
`pyproject.toml`; uv generated the hashed pip lock and is optional for installing the SkyPilot client on your
laptop. Training remains Python 3.10.11 / TensorFlow 2.10.1 / CUDA 11.2 / cuDNN
8.1. The image also includes the pinned DINO source and weights.

The Dockerfile has not yet been built or GPU-tested. Package downloads require
internet access during the build. A build requires several GB of free space.
Use an Ampere GPU such as the previously tested RTX A5000 for this old stack.

## 1. Put the build inputs in Git

The build needs the Dockerfile, `.dockerignore`, `sky/`, `scripts/`, and
`environment/` (excluding ignored validation outputs), plus the checkpoint
repair and W&B project selection in
`hardness_classification/adaptive_sampling.py`. Keep these together when
sharing the deployment commit. Push the commit to your repository, then clone
or pull it on your laptop.

Data, `.venv`, `.cache`, `.tools`, credentials, and existing run outputs belong
outside Git. `.dockerignore` also excludes these from the build context. Do not
use `git add .` to collect unrelated local artifacts.

## 2. Build and publish from the laptop

Install Docker (Docker Desktop on macOS), start it, then run from the repository
root, replacing `YOUR_DOCKERHUB_USER`:

```bash
docker login
docker buildx build --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t YOUR_DOCKERHUB_USER/tactile-hardness:tf210-v1 \
  --push .
```

`linux/amd64` is required even on an Apple Silicon laptop. Emulated builds can
be slow. Use a new tag for each revision (or a registry digest) rather than
overwriting tags used for reported experiments. Git stores the recipe/source;
Docker Hub stores the built image. Start with a public image for simple pulls;
private registries require separate SkyPilot registry credentials.

Optional import check on the laptop (does not require a GPU):

```bash
docker run --rm --platform linux/amd64 \
  YOUR_DOCKERHUB_USER/tactile-hardness:tf210-v1 \
  bash scripts/run.sh python -c 'import tensorflow as tf; print(tf.__version__)'
```

## 3. Configure SkyPilot on the laptop

Use a separate client environment, not the training environment:

```bash
uv venv --python 3.11 .sky-venv
uv pip install --python .sky-venv/bin/python 'skypilot[runpod]' 'runpod>=1.6.1'
source .sky-venv/bin/activate
runpod config
sky check runpod
```

Enter your RunPod API key through `runpod config`. Do not commit it. If uv is
not installed, use `python3.11 -m venv .sky-venv` and install with pip instead.

## 4. Transfer data and launch one training run

Copy the existing processed `data/Own` directory from this container to
`data/Own` in your laptop checkout. It must include `X_4_raw.npy` and
`y_4_raw.npy`. These files are ignored by Git and excluded from the image.
The example uses SkyPilot `file_mounts` to transfer them from your laptop;
large-scale use can replace this with object storage. A volume on the current
RunPod container does not automatically attach to a new SkyPilot pod.

For a regional RunPod Network Volume, SkyPilot also supports registering the
existing volume and attaching it through `volumes`; compute must use the same
datacenter. Global Volumes are a different storage type: native SkyPilot
attachment has not been verified, so do not register one as
`runpod-network-volume`. The checked-in task uses laptop file transfer until
a different attachment method is configured.

Edit `sky/runpod.yaml` to set the image you pushed. Then:

```bash
sky launch -c tactile-train sky/runpod.yaml
sky logs tactile-train
```

This starts **one full ConvLSTM timing run** using the same historical protocol
as the successful benchmark. It is not the replacement quota campaign, does
not add reproducible seeds, and is excluded from the 28-success quota. Do not
launch concurrent jobs in the same pod with the existing shared checkpoint
paths. All code/environment live at `/opt/tactile-hardness` inside the image;
the task explicitly invokes `scripts/run.sh` through the benchmark launcher so
SkyPilot's own Python does not replace the locked training environment.

For the full synthetic GPU/model smoke check before training, replace `run`
in the YAML with:

```yaml
run: |
  cd /opt/tactile-hardness
  bash scripts/run.sh python scripts/verify_environment.py
```

Restore the benchmark command afterward and submit it with `sky exec` to the
same cluster. A successful image build alone does not verify GPU compatibility.

## W&B credentials and online logging

The default is offline logging. To upload new job metrics, export your W&B API
key on the laptop and forward it as a SkyPilot secret. In Bash, this prompts
without echoing the key or putting its value in shell history:

```bash
read -r -s -p 'W&B API key: ' WANDB_API_KEY; echo
export WANDB_API_KEY
sky launch -c tactile-train sky/runpod.yaml \
  --secret WANDB_API_KEY \
  --env WANDB_MODE=online \
  --env WANDB_ENTITY=YOUR_TEAM_OR_USERNAME
unset WANDB_API_KEY
```

Get the key at https://wandb.ai/authorize. It must have access to the selected
team. SkyPilot forwards the shell value with `--secret WANDB_API_KEY` and
redacts it in its dashboard. W&B reads that environment variable directly;
there is no need for `wandb login` inside the image or a copied `.netrc`.
Repeat the secret flag when submitting a later job with `sky exec`.

The task uses `WANDB_PROJECT=tactile-hardness-supplementary`; override it with
`--env WANDB_PROJECT=YOUR_PROJECT` if needed. The driver now honors that
variable; without it, historical prefix-based project naming is preserved.
Keep new reconstruction runs separate from the original archived project.
The benchmark launcher now respects `WANDB_MODE` instead of forcing offline.
Do not bake the key into Docker layers, Git, YAML, or dataset files. Online
metrics do not automatically upload the HDF5 checkpoints; retrieve those below.

Reference: [SkyPilot secret forwarding](https://docs.skypilot.ai/en/v0.11.2/running-jobs/environment-variables.html).

## 5. Retrieve outputs before deleting the pod

SkyPilot configures SSH access under the cluster name. On the laptop:

```bash
mkdir -p results/tactile-train
rsync -av tactile-train:/opt/tactile-hardness/runs/ results/tactile-train/runs/
rsync -av tactile-train:/opt/tactile-hardness/hardness_classification/model_dumps/ results/tactile-train/model_dumps/
rsync -av tactile-train:/opt/tactile-hardness/hardness_classification/AdaptiveModels/ results/tactile-train/AdaptiveModels/
# After checking the downloaded results:
sky down tactile-train
```

Training completion does not terminate the pod. It remains billable until
deleted; download outputs first. Console logs alone do not preserve model
checkpoints or complete local W&B artifacts. For sustained training, configure
durable external output storage before using automatic termination.

References: [SkyPilot Docker runtime requirements](https://docs.skypilot.ai/en/latest/examples/docker-containers.html)
and [RunPod client setup](https://docs.skypilot.ai/en/latest/getting-started/installation.html#runpod).
RunPod uses this image as its pod runtime, not nested `docker run`. The image
uses Ubuntu, root, and a plain shell command as required by that integration.
