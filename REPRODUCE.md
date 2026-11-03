# Reproduce the experiments

Run commands from the repository root unless indicated otherwise. The runner
trains new models for Figures 3–5 and the model results in Table 1. Analysis
and plotting tools are excluded from Git. Random sampling is unseeded, so new
runs reproduce the protocol and populations rather than identical numbers.

## 1. Use the experiment image

`sky/compute.yaml` uses `fheller0/tactile-hardness:tf210-smoke-v1`. Use this
image for the data preparation pod too. It will be public with the repository
release; before then, access requires permission to pull the private image.

The Linux x86_64 image uses Python 3.10.11, TensorFlow 2.10.1 / Keras 2.10.0,
CUDA 11.2 and cuDNN 8.1. Package locks and DINO asset hashes are in
`environment/`. This reconstructs the environment; the original runs used
TensorFlow 2.9 and later ConvLSTM replenishment used 2.10.

To rebuild instead, build and publish from your machine, replacing `YOUR_REGISTRY`:

```bash
docker build --platform linux/amd64 -t YOUR_REGISTRY/tactile-hardness:paper .
docker push YOUR_REGISTRY/tactile-hardness:paper
```

If rebuilding, set `resources.image_id` in `sky/compute.yaml` to your image,
preferably its registry digest. Adjust the GPU, RAM and disk for your data and availability.
For local Linux execution instead, run `bash scripts/setup_environment.sh`
and `bash scripts/run.sh python scripts/verify_environment.py`.

## 2. Populate a RunPod volume from Google Drive

Skip downloading if your volume already contains the four cached arrays below.
Otherwise, create a **Network Volume** in RunPod's **Storage** page. Choose a
data center with a suitable GPU and space for downloads, extracted data,
caches and results. Deploy a temporary Secure Cloud Pod with that volume
mounted at `/workspace`, using the image above and container command
`sleep infinity`. See [RunPod's volume guide](https://docs.runpod.io/storage/network-volumes).

Inside the temporary pod, supply the dataset publisher's Google Drive URL.
The downloader uses a separate environment from the experiment packages:

```bash
df -h /workspace
mkdir -p /workspace/downloads /workspace/staging /workspace/data
bash /opt/tactile-hardness/scripts/run.sh python -m venv /tmp/hardness-download
/tmp/hardness-download/bin/python -m pip install 'gdown==5.2.0'
DATASET_URL='YOUR_GOOGLE_DRIVE_URL'
/tmp/hardness-download/bin/gdown --fuzzy --continue "$DATASET_URL" \
  -O /workspace/downloads/dataset.zip
unzip -n /workspace/downloads/dataset.zip -d /workspace/staging
```

For a Drive folder, replace the download/extraction commands with
`/tmp/hardness-download/bin/gdown --folder "$DATASET_URL" -O /workspace/staging/`.
The link must allow public downloads; restricted data needs the publisher's
authorized transfer procedure. This pinned downloader limits folders to 50
files each, so prefer an archive for larger collections. For `.tar.gz`, use
that extension and extract with `tar -xzf`. See [gdown usage](https://github.com/wkentaro/gdown/tree/v5.2.0).

Copy this repository checkout to `/workspace/tactile-hardness` on the preparation
pod (for example, `git clone YOUR_REPOSITORY_URL /workspace/tactile-hardness`).
Use its current scripts with the image's Python environment, as shown below;
SkyPilot similarly syncs the checkout for training.

The volume must contain:

```text
/workspace/data/Own/X_4_raw.npy
/workspace/data/Own/y_4_raw.npy
/workspace/data/Yuan/FLAT/X_3_raw.npy
/workspace/data/Yuan/FLAT/y_3_raw.npy
```

For downloaded caches, locate the directory containing `Own/` and `Yuan/`
and copy its contents, adjusting the archive wrapper path:

```bash
rsync -av --ignore-existing /workspace/staging/YOUR_CACHE_ROOT/ /workspace/data/
```

For raw recordings instead, preprocess once in the fresh preparation pod:

```bash
cd /workspace/tactile-hardness
ln -s /workspace/data data
bash /opt/tactile-hardness/scripts/run.sh python scripts/reproduce.py prepare \
  --own /workspace/staging/YOUR_OWN_RAW_DIRECTORY \
  --yuan /workspace/staging/YOUR_YUAN_Data1_DIRECTORY
```

Own needs `meta/`, `frames/`, `forces/`; Yuan needs the original `Data1`
directory. Preprocessing uses four-frame Own and three-frame Yuan sequences.
Use the released data without extra normalization and retain its version and
publisher checksums. Existing caches are reused; prepare a different release
in an empty cache directory.

Validate the four paths and connect the preparation checkout to the volume:

```bash
cd /workspace/tactile-hardness
bash /opt/tactile-hardness/scripts/run.sh python scripts/link_storage.py \
  --data-root /workspace/data --results-root /workspace/results
```

The helper checks file presence, not contents, and refuses to replace existing
directories or mismatched links. Terminate the preparation pod and retain the
volume for training.

## 3. Attach the volume to compute

In `sky/runpod-volume.yaml`, set the exact existing volume **name** and
`infra: runpod/YOUR_COUNTRY/YOUR_DATA_CENTER`; keep `use_existing: true`.
Use the same name under `volumes` in `sky/compute.yaml`. If needed, change
`DATA_ROOT` and `RESULTS_ROOT` there; defaults are `/workspace/data` and
`/workspace/results`. Compute must be in the volume's data center. See
[SkyPilot volume configuration](https://docs.skypilot.ai/en/latest/reference/volumes.html#volumes-on-runpod).

On your submitting machine:

```bash
python3 -m venv /tmp/hardness-sky
source /tmp/hardness-sky/bin/activate
pip install 'skypilot[runpod]==0.14.1'
export RUNPOD_API_KEY='YOUR_RUNPOD_API_KEY'
sky check runpod
sky volumes apply sky/runpod-volume.yaml
sky volumes ls -v
bash sky/setup.sh
```

Setup provisions paid compute, attaches the volume, syncs source and checks
the arrays and GPU. It does not start training or upload a laptop dataset.
A volume must be attached when creating the pod; `sky exec` cannot add it
to an existing pod. Use a new cluster name if one already exists without it.

## 4. Run the experiments

Set your W&B entity and API key on the submitting machine, then submit each
job separately, waiting for it to finish before submitting the next:

```bash
export WANDB_ENTITY='YOUR_WANDB_USER_OR_TEAM'
read -rsp 'W&B API key: ' WANDB_API_KEY
export WANDB_API_KEY
sky exec tactile-hardness sky/experiment-model-comparison.yaml \
  --env WANDB_ENTITY --secret WANDB_API_KEY
sky exec tactile-hardness sky/experiment-dropout.yaml \
  --env WANDB_ENTITY --secret WANDB_API_KEY
sky exec tactile-hardness sky/experiment-active-sampling.yaml \
  --env WANDB_ENTITY --secret WANDB_API_KEY
```

| Job | Population |
| --- | --- |
| Model comparison (Figure 3) | 50 attempts per model/dataset; 300 total, including nonconverged attempts. |
| Dropout (Figure 4) | DINOv2NN/Yuan, dropout 0.05–0.50; 90 completed trajectories. |
| Active sampling (Figure 5 and Table 1 model results) | Variance, entropy, random; 150 completed Own and 135 completed Yuan trajectories. |

Initial training is 100 epochs on Own and 50 on Yuan; five updates add 50/25
epochs each. The runner preserves the driver's 40-sample training cap, which
differs from the paper's stated 80 for OFNN/DINOv2NN. Figures 4–5 retry
nonconvergence up to five times each group's completion target; errors or an
unfilled quota stop the job. Rerunning resumes groups and retains failed attempts.

Preview a campaign with `python scripts/reproduce.py active-sampling --dry-run`.
For local training, use `bash scripts/run.sh python scripts/reproduce.py
active-sampling` with the same credentials and a `data/` directory containing
the arrays above. Substitute `model-comparison` or `dropout` as needed.
For offline training, set `WANDB_MODE=offline` locally or pass
`--env WANDB_MODE=offline` to the SkyPilot job.

## 5. Retrieve results and shut down

Attempts, metric histories and stdout are saved under
`/workspace/results/<experiment>/<group>/<attempt>/`. Online W&B runs also
upload `metrics.json`; checkpoint saving is disabled.

```bash
sky queue tactile-hardness
sky logs tactile-hardness JOB_ID
rsync -av tactile-hardness:/workspace/results/ ./downloaded-results/
sky down tactile-hardness
```

Keep the network volume: datasets and results persist there after compute
termination. Reattach it with the same results path to resume; use a new
results directory for an independent campaign or after changing code/data.
