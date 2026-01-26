# LTX-Video Worker (RTX 4090 / Ada sm_89)
# Stable baseline: PyTorch 2.5.1 + CUDA 12.1 + cuDNN9 runtime
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

WORKDIR /app

# Minimal OS deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      curl \
      ca-certificates \
      tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY app/ /app/app/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /workspace/models/hf_home /workspace/models/hf_hub /workspace/models/hf_transformers /workspace/outputs

EXPOSE 8000

ENV PORT=8000 \
    WORKDIR=/workspace \
    OUTPUT_DIR=/workspace/outputs \
    HF_HOME=/workspace/models/hf_home \
    HF_HUB_CACHE=/workspace/models/hf_hub \
    TRANSFORMERS_CACHE=/workspace/models/hf_transformers \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini","--","/app/entrypoint.sh"]
