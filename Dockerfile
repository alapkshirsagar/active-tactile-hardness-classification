# syntax=docker/dockerfile:1
# The package locks and DINO assets target Linux x86_64.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash bzip2 ca-certificates curl git libgl1 libglib2.0-0 \
    openssh-server python3 python3-pip rsync sudo tmux unzip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

# Keep training Python separate from SkyPilot's runtime Python. CUDA/cuDNN
# come from the native lock; the GPU driver is supplied by RunPod.
WORKDIR /opt/tactile-hardness
COPY environment/ environment/
COPY scripts/ scripts/
RUN bash scripts/setup_environment.sh

COPY hardness_classification/ hardness_classification/

ARG VCS_REF=unknown
LABEL org.opencontainers.image.revision=$VCS_REF
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=online

# RunPod/SkyPilot needs a plain shell rather than a training entrypoint.
USER root
ENTRYPOINT []
CMD ["/bin/bash"]
