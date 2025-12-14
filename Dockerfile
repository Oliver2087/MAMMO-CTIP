FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Optional OS libs (often helps image libs; safe to keep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install extras with pip (no conda solve => avoids OOM)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENTRYPOINT ["python", "/app/train_imgTabCLIP_Focal_wFocBCE_resample_full.py"]
CMD ["--help"]