# Active Sampling for Tactile Hardness Exploration

This is the code repository of the *ICRA 2025* paper *Active Sampling for Tactile Hardness Exploration*.

## Installation

For the pinned Linux GPU environment reconstructed and validated against this code, see [environment/README.md](environment/README.md). Run `bash scripts/setup_environment.sh` to recreate it, then `source scripts/env.sh` before running Python. This environment matches the Python/CUDA/cuDNN versions below; other package versions are documented reconstruction choices. For container builds and RunPod jobs, see [Docker and SkyPilot](environment/DOCKER_SKYPILOT.md).

Our project is confirmed to be executable on both *Microsoft Windows 11 [Version 10.0.22631.4037]* and *Ubuntu 20.04.6 LTS*. It should function normally on the *Windows 10*, *Windows 11* and the most *Linux* distributions. We did not test the installation procedure on *WSL*.

First, install the required components of *Tensorflow*: *NVCC* and *CuDNN*.

It's confirmed that the project is fully executable with the following library versions:

* Python 3.10.11
* CUDA 11.2
* cuDNN 8.1.0

Then install the requirement file after `cd` into current directory:

```bash
pip install -r requirements.txt
```

## Training and Evaluation

Please login your *WandB* credential before executing the scripts to make sure that the experiment results can be logged.


First, make sure that you've run `cd hardness_classification`. All scripts must be executed under such folder. Later we omit this folder name as the directory prefix.

Our main executable script is the following:

```bash
python ./leave_one_out.py --comp_index --strat --prefix --model --dataset
```

Keep in mind that you need to change the directory of both datasets accordingly in this python file.

The first run will initially take some time to preprocess the dataset, after that the script will continue to use the cached subsampled data, unless the parameters of dataset manager is changed.

## Experiments

For reproducing our experiment results, you can utilize the *Windows Batch Scripts* or *Linux Bash Scripts* to simplify the calling of `leave_one_out.py` .

### Model Comparison

Strategy *YOTO* is used for disable adaptive sampling.

Please refer to `compare_models.sh` and change the path to *Python* interpreter and the path to project according to your local settings.

### Adaptive Sampling

Please refer to `experiment_iph.bat` and  `experiment_yuan.bat`.

### Dropout Rate Comparison

Please refer to `test_dropout.sh` or `test_dropout.bat`. Change paths according to your local settings.

## Download Experiment Results

Use `export_data.py` to export a CSV file from a *WandB* project.

Use `export_metric_hist.py` to export the historical values in

For example usages, please refer to `export_metric.bat`.

## Data Visualization

Please refer to `visualize.ipynb`. You can change settings and path to exported CSV files in this *Jupyter Notebook* document.

## ConvLSTM replenishment

See [experiments/REPLENISHMENT.md](experiments/REPLENISHMENT.md) and
[sky/replenish-convlstm.yaml](sky/replenish-convlstm.yaml) for the quota-aware
SkyPilot campaign (28 additional successes, no launch performed yet).
