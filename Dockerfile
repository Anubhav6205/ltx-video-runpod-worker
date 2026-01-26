# LTX-Video Worker (RTX 5090 / Blackwell sm_120)
# Use a PyTorch CUDA 12.8 runtime base to avoid "no kernel image" on sm_120.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

WORKDIR /app

# Minimal OS deps (keep image smaller + fewer moving parts)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      curl \
      ca-certificates \
      tini \
    && rm -rf /var/lib/apt/lists/*

# Keep pip tooling modern (but avoid pulling compilers/toolchains)
RUN python -m pip install -U pip setuptools wheel

# Install Python deps (torch is already in the base image)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# Copy app
COPY app/ /app/app/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Runtime dirs (mount volumes here)
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
