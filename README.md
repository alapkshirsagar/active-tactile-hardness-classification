# Active Sampling for Tactile Hardness Exploration

This is the code repository of the *ICRA 2025* paper *Active Sampling for Tactile Hardness Exploration*.

## Installation

Our project is confirmed to be executable on both *Microsoft Windows 11 [Version 10.0.22631.4037]* and *Ubuntu 20.04.6 LTS*. It should function normally on the *Windows 10*, *Windows 11* and the most *Linux* distributions. We did not test the installation procedure on *WSL*.

First, install the required components of *Tensorflow*: *NVCC* and *CuDNN*.

It's confirmed that the project is fully executable with the following library versions:

* Python 3.10.11
* CUDA 11.2
* cuDNN 8.1.0

Then install the requirement file after `cd` into current directory:
```
pip install -r requirements.txt
```


## Usage

### Training and Evaluation

First, make sure that you've run `cd hardness_classification`. All scripts must be executed under such folder.

```
python3 leave_one_out.py
```
Keep in mind that you need to change the directory of both datasets accordingly in this script.

The first run will initially take some time to preprocess the dataset, after that the script will continue to use the cached subsampled data, unless the parameters of dataset manager is changed.

### Experiments

For reproducing our experiment results, you can utilize the *Batch Scripts* or *Bash Scripts* to simplify the calling of `leave_one_out.py` .

