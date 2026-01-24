#!/bin/bash
set -e

echo "=================================================="
echo "LTX-Video Worker - Entrypoint"
echo "=================================================="

# Export HuggingFace cache environment variables
export HF_HOME=${HF_HOME:-/workspace/models/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/workspace/models/hf_hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/workspace/models/hf_transformers}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"

# Create directories if they don't exist
echo "Creating cache directories..."
mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "${OUTPUT_DIR:-/workspace/outputs}"

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found - GPU may not be available!"
fi

# Verify PyTorch CUDA availability
echo ""
echo "Verifying PyTorch CUDA setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('ERROR: CUDA is not available! LTX-Video requires GPU.')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "FATAL: PyTorch CUDA verification failed!"
    exit 1
fi

echo ""
echo "Starting model initialization..."
echo "This may take several minutes on first run (downloading weights)"
echo ""

# Optional: Pre-download model weights before starting server
# This can be disabled if you want lazy loading on first request
if [ "${PRELOAD_MODEL:-true}" = "true" ]; then
    echo "Pre-loading model weights..."
    python3 -c "from app.bootstrap import init_model; init_model()" || {
        echo "WARNING: Model pre-loading failed - will retry on first request"
    }
fi

echo ""
echo "Starting FastAPI server..."
echo "Port: ${PORT:-8000}"
echo "=================================================="

# Start uvicorn server
exec python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --log-level "${LOG_LEVEL:-info}" \
    --no-access-log
