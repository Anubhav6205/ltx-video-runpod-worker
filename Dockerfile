# LTX-Video Worker Dockerfile
# Optimized for NVIDIA GPU (CUDA 12.1) on RunPod
# Base image: NVIDIA CUDA runtime on Ubuntu 22.04

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# This layer is cached unless requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Create directories for models and outputs
# These should be mounted as volumes for persistence
RUN mkdir -p /models/hf_home /models/hf_hub /models/hf_transformers /workspace/outputs

# Expose HTTP port (RunPod proxy will map this)
EXPOSE 8000

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV WORKDIR=/workspace
ENV OUTPUT_DIR=/workspace/outputs
ENV HF_HOME=/models/hf_home
ENV HF_HUB_CACHE=/models/hf_hub
ENV TRANSFORMERS_CACHE=/models/hf_transformers

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
