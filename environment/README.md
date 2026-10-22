# Reconstructed experiment environment

Installed locally in `.venv/` using micromamba 2.3.3 plus pip. Targets **Linux x86_64 with an NVIDIA GPU/driver**. System Python and system CUDA are not modified.

## Use the installed environment

From the repository root:

```bash
source scripts/env.sh
python --version
cd hardness_classification
python leave_one_out.py --help
```

Or run a single command from the repository root:

```bash
bash scripts/run.sh python scripts/verify_environment.py
bash scripts/run.sh jupyter lab
```

In Jupyter, select **Tactile Hardness (Python 3.10 / TF 2.10)**. Launch Jupyter through the wrapper so kernels inherit the CUDA library path and local DINO cache. Merely adding `.venv/bin` to PATH is insufficient for the GPU libraries. The source script also enables TensorFlow GPU memory growth and defaults to 8 intra-op/OpenMP threads and 2 inter-op threads; set these thread variables before sourcing to override them. No numerical determinism is promised by this environment.

For Docker builds and SkyPilot/RunPod launch instructions, see [DOCKER_SKYPILOT.md](DOCKER_SKYPILOT.md). Dataset paths and experimental protocol still need to be configured for each job. This environment does not alter the sampler or model architecture: the historical driver remains unseeded and can reuse recordings across training and evaluation. Robot collection dependencies (`frankx`, GelSight SDK, robot firmware/configuration) are a separate hardware setup and are not required for offline classification.

## Versions and degree of historical agreement

| Component | Installed | Basis |
|---|---|---|
| Python | 3.10.11 | Exact version stated in README |
| CUDA runtime | 11.2.2 | README specifies CUDA 11.2 |
| cuDNN | 8.1.0.77 | README specifies cuDNN 8.1.0 |
| TensorFlow / Keras | 2.10.1 / 2.10.0 | Reconstruction: compatible with the recorded CUDA/cuDNN stack and existing Keras/private APIs |
| NumPy / SciPy | 1.23.5 / 1.10.1 | Reconstruction: compatible with old TensorFlow and preprocessing |
| PyTorch / torchvision | 2.0.1+cpu / 0.15.2+cpu | Reconstruction: matched official pair; existing DINO wrapper explicitly uses CPU |
| pandas / matplotlib / scikit-learn | 1.5.3 / 3.7.5 / 1.3.2 | Reconstruction, verified with the repository |
| OpenCV / W&B | 4.8.1.78 / 0.17.9 | Reconstruction, verified with the repository |

The original repo did not pin these Python dependencies. Its saved September 2024 CUDA notebook even reports Python 3.8.19 rather than the README's 3.10.11. This is a **working, fully pinned reconstruction**, not proof of the exact historical training environment. Supporting native libraries and transitive Python dependencies were resolved at installation time and then locked; they are not claimed to be 2024 versions.

TensorFlow's [tested build configurations](https://www.tensorflow.org/install/source#gpu) identify CUDA 11.2/cuDNN 8.1 for the TensorFlow 2.10 generation. The PyTorch versions follow its [official matching table](https://pytorch.org/get-started/previous-versions/#v201). The NVIDIA driver is supplied by the host, not the lock; validation used driver 580.159.04 and an RTX A5000. The CUDA version displayed by `nvidia-smi` is the driver's supported version, not the CUDA runtime this environment loads.

## Recreate

```bash
bash scripts/setup_environment.sh
bash scripts/run.sh python scripts/verify_environment.py
```

`setup_environment.sh` downloads native packages and Python packages, prepares DINO, and installs a kernel under `.venv/share/jupyter/`. It is safe to rerun for this environment. A clean checkout is preferred when recreating from scratch. Expected initial downloads include about 1.3 GB of native CUDA/cuDNN packages, the TensorFlow/PyTorch wheels, and the 84 MB DINO checkpoint; extracted caches require additional disk space.

Lock files:

* `conda-linux-64.lock.txt`: explicit native package URLs/builds with checksums, including Python and GPU runtimes.
* `requirements-linux-64.lock.txt`: exact Python dependencies with artifact hashes; installed with `pip --require-hashes`. The conda base's pip/setuptools/wheel/packaging metadata is superseded where specified by this second stage.
* `requirements.in`: intentional top-level reconstruction choices; not the full lock.
* `installed-python.txt`: portable snapshot of actual installed Python versions, used to generate the hashed lock.
* `assets.sha256`: micromamba executable, DINO source archive, and official pretrained-weight hashes.

The original unpinned root `requirements.txt` is retained for provenance. Use the setup script and locks for this reconstruction.

## DINO source and weights

The local `TORCH_HOME` is `.cache/torch`. The original `models.py` requests the mutable default branch of `facebookresearch/dinov2`. To preserve the model code, setup populates its expected `facebookresearch_dinov2_main` cache directory with fixed source revision **e1277af2ba9496fbadf7aec6eba56e8d882d1e35** (2024-02-22, latest upstream commit before 2024-09-05). This date-based selection is a reconstruction choice, not a recovered original commit.

The checkpoint is the official [DINOv2 ViT-S/14 pretrained weight](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth), SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. Source archive and weight hashes are checked by setup. Keep the local cache, use `scripts/env.sh`, and do not call hub with `force_reload=True`; clearing/reloading that cache would lose this pin. Rerunning setup restores the archived source into the expected cache directory. No DINO training extras (xFormers, cuML, submitit) are needed for CPU feature inference.

## Validation

On 2026-10-19, the smoke script passed with synthetic **240×320** videos: four frames for ConvLSTM/OFNN and three frames for DINOv2NN. For each actual repository wrapper it built the model, trained one epoch, checked finite predictions and explicit dropout inference, and verified equivalent predictions after HDF5 save/reload. TensorFlow model weights were on `/GPU:0`; DINO's frozen backbone was on CPU.

It also verified a GPU matrix multiplication, loaded runtime CUDA version `11020` and cuDNN version `8100`, exercised Savitzky–Golay plotting/PDF export and offline W&B logging, and ran `pip check`. The training CLI `--help` imported successfully. Local machine-readable results are in `validation-output/smoke.json` (ignored by git).

TensorFlow 2.10 can print an old cuBLAS registration message and warnings about absent TensorRT. Standard model training and inference passed despite those messages; TensorRT is not needed by these models. This validation does not claim dataset-level reproduction. The adaptive driver's final-save expression was subsequently repaired for the real-data timing benchmarks. One complete ConvLSTM/own-data benchmark took about 26m40s, and one DINO/Yuan benchmark took about 4m26s on the tested RTX A5000 host. These timings describe single runs; the Docker image still needs separate build and GPU validation.
