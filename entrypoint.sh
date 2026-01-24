#!/bin/bash
set -e

echo "=================================================="
echo "LTX-Video Worker - Entrypoint"
echo "=================================================="

# Export HuggingFace cache environment variables
export HF_HOME=${HF_HOME:-/models/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/models/hf_hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/models/hf_transformers}
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
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --log-level "${LOG_LEVEL:-info}" \
    --no-access-log
