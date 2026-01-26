# LTX-Video Worker Dockerfile
# Optimized for NVIDIA GPU (CUDA 12.8) on RunPod - RTX 5090 Ready
# Base image: NVIDIA CUDA runtime on Ubuntu 24.04

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
# Allow pip to modify system packages (PEP 668 on Ubuntu 24.04)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory
WORKDIR /app

# Install system dependencies
# Ubuntu 24.04 includes Python 3.12 by default
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    wget \
    && apt-get purge -y python3-wheel python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.8 support for RTX 5090
# Installed separately to ensure correct CUDA version
# Using 2.7.0 (supports Blackwell/sm_120)
RUN python3 -m pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining Python dependencies
# This layer is cached unless requirements.txt changes
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Create directories for models and outputs
# These should be mounted as volumes for persistence
RUN mkdir -p /workspace/models/hf_home /workspace/models/hf_hub /workspace/models/hf_transformers /workspace/outputs

# Expose HTTP port (RunPod proxy will map this)
EXPOSE 8000

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV WORKDIR=/workspace
ENV OUTPUT_DIR=/workspace/outputs
ENV HF_HOME=/workspace/models/hf_home
ENV HF_HUB_CACHE=/workspace/models/hf_hub
ENV TRANSFORMERS_CACHE=/workspace/models/hf_transformers

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
