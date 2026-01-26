# Docker Deployment Guide for RunPod RTX 4090

This guide covers building and deploying the LTX-Video Worker on RunPod.

## Prerequisites

- Docker installed locally
- NVIDIA GPU with CUDA support (for local testing)
- RunPod account
- Docker registry account (Docker Hub, GitHub Container Registry, etc.)

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t ltx-video-worker:latest .
```

**Expected build time**: 10-15 minutes (first build)

### 2. Test Locally (Optional - requires NVIDIA GPU)

```bash
# Test with GPU
docker run --rm --gpus all ltx-video-worker:latest python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run the full test suite
bash docker-build-test.sh
```

### 3. Push to Registry

```bash
# Tag for your registry
docker tag ltx-video-worker:latest your-username/ltx-video-worker:latest

# Push
docker push your-username/ltx-video-worker:latest
```

### 4. Deploy on RunPod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click "Deploy" → "Custom Template"
3. Configure:
   - **Container Image**: `your-username/ltx-video-worker:latest`
   - **GPU Type**: RTX 4090 (24GB VRAM)
   - **Container Disk**: 20GB minimum
   - **Volume Disk**: 50GB recommended (for model caching)
   - **Expose HTTP Ports**: `8000`
   - **Environment Variables**:
     ```
     PORT=8000
     LOG_LEVEL=INFO
     STORAGE_BACKEND=r2
     R2_BUCKET=your-bucket-name
     R2_ACCOUNT_ID=your-cloudflare-account-id
     R2_ACCESS_KEY_ID=...
     R2_SECRET_ACCESS_KEY=...
     PRELOAD_MODEL=true
     ```

4. Mount volumes:
   - `/workspace` → Network Volume (contains `models/` and `outputs/`)

5. Click "Deploy"

## Docker Image Details

### Base Image
- **nvidia/cuda:12.1.1-runtime-ubuntu22.04**
- Includes CUDA 12.1 runtime for RTX 4090
- Ubuntu 22.04 LTS base

### Python Environment
- **Python**: 3.10
- **PyTorch**: 2.10.0+cu121 (CUDA 12.1)
- **Diffusers**: 0.36.0
- **Transformers**: 4.57.6

### Key Features
1. **CUDA 12.1 Support**: Pre-built PyTorch with RTX 4090 optimization
2. **Model Caching**: Persistent storage for HuggingFace weights
3. **Health Checks**: Built-in health endpoint monitoring
4. **Logging**: Structured JSON logs for production monitoring
5. **Concurrency Control**: Prevents GPU OOM with semaphore limiting

### Directory Structure
```
/app/                   # Application code
  ├── app/             # Python package
  ├── entrypoint.sh    # Startup script
  └── requirements.txt # Dependencies

/workspace/            # Network Volume Mount Point
  ├── models/          # Model cache
  │   ├── hf_home/
  │   ├── hf_hub/
  │   └── hf_transformers/
  ├── outputs/         # Generated videos
  └── gcs-service-account.json # (Optional, only if using GCS)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | HTTP server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PRELOAD_MODEL` | `true` | Download model weights on startup |
| `MAX_CONCURRENT_JOBS` | `1` | Max simultaneous generations (1 for single GPU) |
| `STORAGE_BACKEND` | `r2` | Storage backend (`r2` or `gcs`) |
| `STORAGE_KEY_PREFIX` | `clips` | Object key prefix/folder |
| `R2_BUCKET` | `""` | Cloudflare R2 bucket name |
| `R2_ACCOUNT_ID` | `null` | Cloudflare account ID (or set `R2_ENDPOINT_URL`) |
| `R2_ENDPOINT_URL` | `null` | R2 endpoint URL override |
| `R2_ACCESS_KEY_ID` | `null` | R2 access key ID |
| `R2_SECRET_ACCESS_KEY` | `null` | R2 secret access key |
| `R2_REGION` | `auto` | S3 region (use `auto` for R2) |
| `R2_PUBLIC_BASE_URL` | `null` | Public base URL (supports `{bucket}`) |
| `R2_PRESIGN_TTL_S` | `3600` | Signed URL TTL (seconds) |
| `GCS_BUCKET` | `""` | Google Cloud Storage bucket name (when using `STORAGE_BACKEND=gcs`) |
| `GCS_SERVICE_ACCOUNT_KEY` | `null` | Path to GCS credentials JSON |
| `HF_TOKEN` | `null` | HuggingFace API token (optional) |
| `TORCH_DTYPE` | `bfloat16` | Model precision (bfloat16, float16, float32) |

## API Endpoints

### Health Check
```bash
GET /health
Response: {"ok": true}
```

### Readiness Check
```bash
GET /ready
Response: {"ready": true, "model_loaded": true}
```

### Generate Video
```bash
POST /generate_clip
Content-Type: application/json

{
  "image_url": "https://example.com/image.jpg",
  "video_prompt": "A cat walking in the park",
  "duration_s": 5,
  "width": 704,
  "height": 512,
  "fps": 24,
  "steps": 8,
  "guidance_scale": 1.0
}
```

## Troubleshooting

### Build Issues

**Problem**: PyTorch CUDA build fails
```bash
# Solution: Clear Docker cache and rebuild
docker builder prune
docker build --no-cache -t ltx-video-worker:latest .
```

**Problem**: Out of disk space during build
```bash
# Solution: Clean up Docker
docker system prune -a
```

### Runtime Issues

**Problem**: CUDA not available in container
```bash
# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Check PyTorch build
docker run --rm --gpus all ltx-video-worker:latest python3 -c "import torch; print(torch.__version__)"
# Should show: 2.10.0+cu121
```

**Problem**: Model download fails
```bash
# Check HuggingFace connectivity
docker run --rm ltx-video-worker:latest curl -I https://huggingface.co

# Increase timeout
docker run -e GENERATE_TIMEOUT_S=600 ltx-video-worker:latest
```

**Problem**: Out of VRAM
```bash
# Reduce concurrent jobs
docker run -e MAX_CONCURRENT_JOBS=1 ltx-video-worker:latest

# Use lower precision
docker run -e TORCH_DTYPE=float16 ltx-video-worker:latest
```

## Performance Optimization

### Model Caching & Output Storage
Mount `/workspace` as a persistent network volume. This handles both model weights and video outputs:

```bash
docker run -v my-network-volume:/workspace ltx-video-worker:latest
```

### Network Volumes (RunPod)
Use a single RunPod network volume for best performance:
- **Mount Point**: `/workspace`
- **Size**: 50GB+ recommended
- **Content**: Automatically holds `models/` (cache) and `outputs/` (videos)

## Monitoring

### Logs
```bash
# View container logs
docker logs <container-id>

# Stream logs
docker logs -f <container-id>
```

### Health Monitoring
```bash
# Check health endpoint
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready
```

### GPU Monitoring
```bash
# Inside container
docker exec <container-id> nvidia-smi

# Watch GPU usage
watch -n 1 'docker exec <container-id> nvidia-smi'
```

## Cost Optimization

### RunPod Pricing (RTX 4090)
- **On-Demand**: ~$0.69/hour
- **Spot**: ~$0.34/hour (50% savings, but can be interrupted)

### Recommendations
1. Use **Spot Instances** for non-critical workloads
2. Enable **Auto-Stop** when idle
3. Use **Network Volumes** to avoid re-downloading models
4. Pre-load model on startup to reduce first-request latency

## Production Checklist

- [ ] Model weights cached in persistent volume
- [ ] R2 credentials configured (or GCS if using `STORAGE_BACKEND=gcs`)
- [ ] Health checks enabled
- [ ] Logging level set appropriately (INFO or WARNING)
- [ ] Auto-scaling configured (if needed)
- [ ] Monitoring/alerting set up
- [ ] Backup/disaster recovery plan
- [ ] Cost alerts configured

## Support

For issues specific to:
- **Docker build**: Check build logs and verify CUDA compatibility
- **RunPod deployment**: Contact RunPod support
- **LTX-Video model**: See [official repo](https://github.com/Lightricks/LTX-Video)
- **Application bugs**: Check application logs at `/workspace/outputs/logs/`
