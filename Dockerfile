# GPU runtime base (CUDA 12.1 + cuDNN8 on Ubuntu 22.04)
FROM nvidia/cuda:11.8.1-cudnn8-runtime-ubuntu22.04

# ---- OS deps (common for torch/torchvision/opencv/PIL + building pip wheels) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bzip2 git \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# ---- micromamba ----
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

WORKDIR /app

# Copy env spec first for Docker layer caching
COPY environment.yml /tmp/environment.yml

# Create conda env named "env" from environment.yml
RUN micromamba create -y -n env -f /tmp/environment.yml \
 && micromamba clean --all --yes

# Use the env by default (so "python" is from /opt/conda/envs/env/bin)
ENV PATH=/opt/conda/envs/env/bin:/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy the rest of your project
COPY . /app

# (optional) If your repo includes run scripts
RUN chmod +x /app/run_train.sh /app/run_eval.sh 2>/dev/null || true

# Default to running your training/eval script; override args at `docker run ...`
# Change this filename to your actual entry script if different.
ENTRYPOINT ["python", "/app/train_imgTabCLIP_Focal_wFocBCE_resample_full.py"]
CMD ["--help"]