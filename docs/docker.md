# Training image

The checked-in Dockerfile and deployment files now use the tested environment
added in commit c56cda4. This supersedes the earlier uv.lock-based scaffold.

Follow [environment/DOCKER_SKYPILOT.md](../environment/DOCKER_SKYPILOT.md) for
building and publishing the image, configuring SkyPilot, W&B secret injection,
GPU verification, and retrieving outputs. See
[environment/README.md](../environment/README.md) for environment provenance.

The setup uses Python 3.10.11, TensorFlow 2.10.1, CUDA 11.2, and cuDNN 8.1,
with explicit native-package and hashed pip locks. There is no requirement for
pyproject.toml or uv.lock. The host supplies the NVIDIA driver.

Current deployment scope:
- sky/runpod.yaml requests one A40 and launches one ConvLSTM timing run.
- Dataset arrays are read from an existing Network Volume mounted at /workspace.
  Fill in its exact name and region/data-centre in sky/runpod-volume.yaml.
- The seeded replenishment campaign must be configured separately before
  launching the remaining runs.
- The Dockerfile has not yet been built or GPU-tested in this workspace.
