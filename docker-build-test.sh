#!/bin/bash
# Docker Build & Test Script for LTX-Video Worker
# Verifies that the Docker image builds correctly and runs

set -e

IMAGE_NAME="ltx-video-worker"
IMAGE_TAG="latest"

echo "=========================================="
echo "LTX-Video Worker - Docker Build Test"
echo "=========================================="

# Step 1: Build the image
echo ""
echo "Step 1: Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi
echo "✅ Docker build successful!"

# Step 2: Verify image exists
echo ""
echo "Step 2: Verifying image..."
docker images | grep ${IMAGE_NAME}

# Step 3: Test PyTorch CUDA in container
echo ""
echo "Step 3: Testing PyTorch CUDA availability..."
docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG} python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    assert 'cu121' in torch.__version__, 'Expected CUDA 12.1 build'
    print('✅ CUDA check passed!')
else:
    print('❌ CUDA not available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch CUDA test failed!"
    exit 1
fi

# Step 4: Test imports
echo ""
echo "Step 4: Testing package imports..."
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python3 -c "
import sys
print('Python version:', sys.version)

# Test all critical imports
from app.settings import settings
from app.models import GenerateClipRequest
from app.bootstrap import get_torch_dtype
import torch
import diffusers
import transformers
import accelerate
import huggingface_hub

print('✅ All imports successful!')
print('Packages:')
print('  - torch:', torch.__version__)
print('  - diffusers:', diffusers.__version__)
print('  - transformers:', transformers.__version__)
print('  - huggingface_hub:', huggingface_hub.__version__)
"

if [ $? -ne 0 ]; then
    echo "❌ Import test failed!"
    exit 1
fi

# Step 5: Image size check
echo ""
echo "Step 5: Checking image size..."
IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${IMAGE_TAG} --format "{{.Size}}")
echo "Image size: ${IMAGE_SIZE}"

echo ""
echo "=========================================="
echo "✅ All tests passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Tag the image: docker tag ${IMAGE_NAME}:${IMAGE_TAG} your-registry/${IMAGE_NAME}:${IMAGE_TAG}"
echo "2. Push to registry: docker push your-registry/${IMAGE_NAME}:${IMAGE_TAG}"
echo "3. Deploy on RunPod with GPU support"
echo ""
