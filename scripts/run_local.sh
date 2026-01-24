#!/bin/bash
# Local development runner for LTX-Video Worker
# Runs the FastAPI server with hot-reload enabled

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    echo "Please copy .env.example to .env and configure it:"
    echo "  cp .env.example .env"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found - GPU may not be available"
else
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"

echo "Starting LTX-Video Worker (development mode)"
echo "============================================="
echo "Port: ${PORT:-8000}"
echo "Output directory: $OUTPUT_DIR"
echo "HF cache: $HF_HUB_CACHE"
echo ""

# Run with hot-reload
python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --reload \
    --log-level "${LOG_LEVEL:-info}"
