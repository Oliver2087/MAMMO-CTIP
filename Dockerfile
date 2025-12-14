FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# OS libs that help opencv/PIL in many cases
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

# Install only "extra" deps (NOT pytorch / pytorch-cuda)
COPY environment.docker.yml /tmp/environment.docker.yml
RUN conda env update -n base -f /tmp/environment.docker.yml && conda clean -a -y

COPY . /app

ENTRYPOINT ["python", "/app/train_imgTabCLIP_Focal_wFocBCE_resample_full.py"]
CMD ["--help"]