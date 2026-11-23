# LTX-Video RunPod Worker

A FastAPI service for generating videos from images using **LTX-Video 2B distilled model**. Designed for deployment on RunPod GPU pods (RTX 4090) with Cloudflare R2 (default) or Google Cloud Storage (GCS) uploads.

## Features

- **Image-to-Video Generation** using LTX-Video 2B distilled model
- **REST API** with FastAPI (OpenAPI docs at `/docs`)
- **Cloudflare R2 Uploads (default)** for video storage and URL generation
- **Optional GCS Uploads** for legacy setups
- **Concurrency Control** to prevent VRAM OOM on single GPU
- **Structured Logging** with request/job ID tracking
- **Docker Support** with NVIDIA GPU runtime
- **HuggingFace Caching** for persistent model weights
- **Input Validation** and safety limits
- **Health & Readiness** endpoints for monitoring

## API Overview

### Endpoints

- `GET /health` - Health check (always returns 200 if server is running)
- `GET /ready` - Readiness check (returns 200 only when model is loaded)
- `POST /generate_clip` - Generate video from image + prompt
- `POST /test_r2` - Upload an MP4 to R2 (verifies R2 credentials + URL generation)
- `GET /` - Service info and endpoint list

### Example Request

```bash
curl -X POST http://localhost:8000/generate_clip \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/input-frame.png",
    "video_prompt": "Cinematic slow dolly-in, hair blowing in wind, natural light",
    "duration_s": 5.0,
    "width": 704,
    "height": 512,
    "fps": 24,
    "steps": 8,
    "seed": 1234
  }'
```

### Example Response

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "clip_url": "https://<your-public-base>/clips/f47ac10b-58cc-4372-a567-0e02b2c3d479.mp4",
  "meta": {
    "fps": 24,
    "num_frames": 121,
    "width": 704,
    "height": 512,
    "steps": 8,
    "model": "ltxv-2b-distilled",
    "seed": 1234,
    "guidance_scale": 1.0,
    "duration_s": 5.0
  }
}
```

## Local Development

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+
- 16GB+ VRAM (24GB recommended)
- ffmpeg installed
- Cloudflare R2 bucket and credentials (default) OR Google Cloud Storage bucket and credentials (optional)

### Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd ltx-worker
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Cloudflare R2 (default)**

Create an R2 bucket and an R2 API token (access key + secret) in Cloudflare, then set these in `.env`:

```bash
STORAGE_BACKEND=r2
R2_BUCKET=your-bucket-name
R2_ACCOUNT_ID=your-cloudflare-account-id
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
```

5. **Configure environment**

```bash
cp .env.example .env
# Edit .env and fill in your configuration:
# - R2_* variables (default) OR set STORAGE_BACKEND=gcs and configure GCS_*
# - Optional: HF_TOKEN for private models
# - Optional: Adjust paths, timeouts, model settings
```

6. **Run the server**

```bash
# Using the helper script
chmod +x scripts/run_local.sh
./scripts/run_local.sh

# Or manually
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

7. **Test the API**

Visit `http://localhost:8000/docs` for interactive API documentation.

```bash
# Check health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready

# Generate a video (replace with your image URL)
curl -X POST http://localhost:8000/generate_clip \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/test.jpg", "video_prompt": "A person walking"}'
```

### Test R2 Upload

```bash
# Upload an existing MP4 from the worker filesystem (must be under WORKDIR/OUTPUT_DIR by default)
curl -X POST http://localhost:8000/test_r2 \
  -H "Content-Type: application/json" \
  -d '{"mp4_path": "/workspace/outputs/example.mp4"}'
```

If your MP4 is outside `WORKDIR`/`OUTPUT_DIR`, set `TEST_R2_ALLOWED_ROOTS` (or `TEST_R2_ALLOW_ANY_PATH=true`, unsafe) in your `.env`.

## Docker Deployment

### Build the Image

```bash
# Build the Docker image
docker build -t your-registry/ltx-worker:0.1 .

# For multi-platform builds (optional)
docker buildx build --platform linux/amd64 -t your-registry/ltx-worker:0.1 .
```

### Run Locally with Docker

```bash
# Run with GPU support
docker run --gpus all \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  -v $(pwd)/outputs:/workspace/outputs \
  your-registry/ltx-worker:0.1
```

### Push to Registry

```bash
# Login to your registry (Docker Hub, GHCR, etc.)
docker login

# Push the image
docker push your-registry/ltx-worker:0.1
```

## RunPod Deployment

### Prerequisites

1. **Push your Docker image** to a public registry (Docker Hub, GHCR, etc.)
2. **Configure Cloudflare R2** for video uploads (default) by setting `R2_*` environment variables
3. *(Optional)* Configure GCS by setting `STORAGE_BACKEND=gcs` and `GCS_*` variables

### Step-by-Step Deployment

1. **Go to RunPod Dashboard**
   - Navigate to https://runpod.io/console/pods

2. **Create a New Pod**
   - Click "Deploy" → "GPU Pod"
   - Select GPU: **RTX 4090** (24GB VRAM recommended)
   - Select a region with availability

3. **Configure the Pod**
   - **Docker Image**: `your-registry/ltx-worker:0.1`
   - **Container Disk**: 50GB minimum (for model weights)
   - **Volume**: Mount a persistent volume at `/models` (50GB+) to cache model weights between pod restarts
   - **Expose HTTP Ports**: `8000`
   - **Environment Variables**: Add all variables from your `.env`:
      ```
      STORAGE_BACKEND=r2
      R2_BUCKET=your-bucket-name
      R2_ACCOUNT_ID=your-cloudflare-account-id
      R2_ACCESS_KEY_ID=...
      R2_SECRET_ACCESS_KEY=...
      HF_TOKEN=your-hf-token (optional)
      WARMUP_ON_START=true (optional)
      ```
   - *(Optional)* If using GCS instead: set `STORAGE_BACKEND=gcs` and configure `GCS_*` variables

4. **Deploy and Wait**
   - Click "Deploy On-Demand" or "Deploy Spot"
   - Wait for pod to initialize (2-5 minutes)
   - Check logs for "Server ready to accept requests"

5. **Access Your Worker**
   - RunPod will provide a proxy URL in the format:
     ```
     https://<pod-id>-8000.proxy.runpod.net
     ```
   - Test health: `https://<pod-id>-8000.proxy.runpod.net/health`
   - Test readiness: `https://<pod-id>-8000.proxy.runpod.net/ready`
   - API docs: `https://<pod-id>-8000.proxy.runpod.net/docs`

### RunPod Tips

- **Persistent Volume**: Always mount a volume at `/models` to avoid re-downloading model weights on every pod restart (saves time and bandwidth)
- **Spot Instances**: Use spot instances for cost savings (can be interrupted but much cheaper)
- **Auto-scaling**: For production, consider RunPod Serverless for automatic scaling
- **Monitoring**: Check pod logs regularly during first few runs to ensure model loads correctly
- **R2 Credentials**: Store access key/secret securely (RunPod secrets recommended)

## Configuration Reference

### Environment Variables

See [.env.example](.env.example) for all available configuration options.

#### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | HTTP server port |
| `MAX_CONCURRENT_JOBS` | `1` | Max simultaneous generations (1 = safe for single GPU) |
| `LTX_REPO_ID` | `Lightricks/LTX-Video` | HuggingFace model repository |
| `TORCH_DTYPE` | `bfloat16` | Precision (bfloat16/float16/float32) |
| `DEFAULT_STEPS` | `8` | Inference steps (4-8 optimal for distilled) |
| `DEFAULT_GUIDANCE_SCALE` | `1.0` | Guidance scale (1.0 for distilled) |
| `STORAGE_BACKEND` | `r2` | Storage backend (`r2` or `gcs`) |
| `STORAGE_KEY_PREFIX` | `clips` | Object key prefix/folder |
| `R2_BUCKET` | *required* | Cloudflare R2 bucket name |
| `R2_ACCOUNT_ID` | *required* | Cloudflare account ID (or set `R2_ENDPOINT_URL`) |
| `R2_ENDPOINT_URL` | *optional* | R2 endpoint URL override |
| `R2_ACCESS_KEY_ID` | *required* | R2 access key ID |
| `R2_SECRET_ACCESS_KEY` | *required* | R2 secret access key |
| `R2_REGION` | `auto` | S3 region (use `auto` for R2) |
| `R2_PUBLIC_BASE_URL` | *optional* | Public base URL (supports `{bucket}`) |
| `R2_PRESIGN_TTL_S` | `3600` | Signed URL TTL (seconds) |
| `GCS_BUCKET` | *optional* | GCS bucket name (when using `STORAGE_BACKEND=gcs`) |
| `HF_TOKEN` | *optional* | HuggingFace token (for private models) |

### Default Generation Parameters

The following defaults are optimized for LTX-Video 2B distilled:

- **Width**: 704px
- **Height**: 512px
- **FPS**: 24
- **Steps**: 8 (distilled model works well with 4-8 steps)
- **Guidance Scale**: 1.0 (distilled model is trained for low guidance)
- **Duration**: 5.0 seconds

### Safety Limits

Built-in safety limits prevent abuse:

- Max width/height: 1280px
- Max FPS: 30
- Max duration: 10 seconds
- Max steps: 30

## Storage Configuration

### Cloudflare R2 Setup (Default)

1. Create an R2 bucket in the Cloudflare dashboard.
2. Create an R2 API token (S3 credentials) with read/write access to the bucket.
3. Configure `.env`:

```bash
STORAGE_BACKEND=r2
R2_BUCKET=your-bucket-name
R2_ACCOUNT_ID=your-cloudflare-account-id
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
```

### URL Generation Options

The worker supports two methods for generating video URLs:

#### 1. Public Base URL (Recommended)

```bash
# .env configuration for public URL
R2_PUBLIC_BASE_URL=https://<your-public-base>
```

Returns: `https://<your-public-base>/clips/{job_id}.mp4`

#### 2. Signed URLs (Default when `R2_PUBLIC_BASE_URL` is empty)

```bash
R2_PRESIGN_TTL_S=3600
```

Returns: a presigned `GET` URL (expires after TTL seconds).

### Google Cloud Storage Setup (Optional/Legacy)

If you need to keep using GCS, set `STORAGE_BACKEND=gcs` and configure `GCS_BUCKET` (and optionally `GCS_SERVICE_ACCOUNT_KEY`, `GCS_PUBLIC_BASE`, etc.).

## Troubleshooting

### Model Not Loading

**Symptom**: `/ready` returns `{"ready": false}`

**Solutions**:
- Check logs for HuggingFace download errors
- Verify GPU is available: `nvidia-smi`
- Ensure sufficient VRAM (16GB minimum)
- Check HF_TOKEN if using private models
- Verify internet connectivity for model download

### Out of Memory (OOM)

**Symptom**: CUDA OOM errors during generation

**Solutions**:
- Reduce `MAX_CONCURRENT_JOBS` to 1
- Use smaller resolution (e.g., 512x512)
- Reduce `steps` parameter
- Use `torch_dtype=float16` instead of bfloat16
- Check for memory leaks (restart pod)

### Slow Generation

**Symptom**: Generations take >60 seconds

**Solutions**:
- First generation is always slower (model compilation)
- Enable warmup: `WARMUP_ON_START=true`
- Reduce `steps` (try 4-6 for distilled model)
- Check GPU utilization: `nvidia-smi dmon`
- Ensure CUDA is being used (check logs)

### R2 Upload Failures

**Symptom**: Generation succeeds but upload fails

**Solutions**:
- Verify `R2_BUCKET`, `R2_ACCOUNT_ID` (or `R2_ENDPOINT_URL`), `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY`
- Ensure the R2 API token has permission to read/write the bucket
- If using `R2_PUBLIC_BASE_URL`, verify your public domain/CDN is correctly configured
- If using `STORAGE_BACKEND=gcs`, follow the GCS setup and credential troubleshooting steps

### 429 Too Many Requests

**Symptom**: HTTP 429 when submitting jobs

**Cause**: Concurrency limit reached

**Solutions**:
- Wait for current job to complete
- Increase `MAX_CONCURRENT_JOBS` (if GPU has enough VRAM)
- Implement request queuing on client side

## Performance Optimization

### Model Compilation (First Run)

The first generation will be slower (~2-3x) due to:
- CUDA kernel compilation
- cuDNN autotuning
- Memory allocation

**Fix**: Enable warmup on startup:
```bash
WARMUP_ON_START=true
```

### Persistent Caching

Mount a persistent volume at `/models` to avoid re-downloading:

```bash
docker run --gpus all \
  -v /path/to/persistent/storage:/models \
  ...
```

On RunPod, create a network volume and mount it at `/models`.

### Batch Processing

For processing multiple images:
- Submit requests sequentially (single GPU limitation)
- Use multiple RunPod pods for parallel processing
- Consider RunPod Serverless for auto-scaling

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Client                        │
└─────────────────┬───────────────────────────────┘
                  │ POST /generate_clip
                  │
┌─────────────────▼───────────────────────────────┐
│              FastAPI Server                     │
│  ┌──────────────────────────────────────────┐  │
│  │  Request Validation (Pydantic)           │  │
│  │  Concurrency Control (Semaphore)         │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                             │
│  ┌────────────────▼─────────────────────────┐  │
│  │  Image Download (requests + PIL)         │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                             │
│  ┌────────────────▼─────────────────────────┐  │
│  │  LTX-Video Pipeline (Diffusers)          │  │
│  │  - Load LTXImageToVideoPipeline           │  │
│  │  - Generate video frames                 │  │
│  │  - Export to MP4 (imageio-ffmpeg)        │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                             │
│  ┌────────────────▼─────────────────────────┐  │
│  │  GCS Upload (google-cloud-storage)       │  │
│  │  - Upload MP4                            │  │
│  │  - Generate public/signed URL            │  │
│  └────────────────┬─────────────────────────┘  │
│                   │                             │
│  ┌────────────────▼─────────────────────────┐  │
│  │  Response (JSON with clip_url)           │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Model Information

- **Model**: LTX-Video 2B distilled
- **Repository**: [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- **Type**: Image-to-video diffusion model
- **Weights**: `ltxv-2b-0.9.8-distilled.safetensors`
- **Precision**: bfloat16 (default)
- **VRAM**: ~12GB for 704x512 @ 24fps
- **License**: Check HuggingFace model card

## License

This worker implementation is provided as-is. Please check the LTX-Video model license before commercial use.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review RunPod documentation: https://docs.runpod.io
- Check Diffusers documentation: https://huggingface.co/docs/diffusers
- Review GCS documentation: https://cloud.google.com/storage/docs

## Credits

- **LTX-Video Model**: Lightricks
- **Diffusers Library**: HuggingFace
- **Compute Platform**: RunPod
- **Storage**: Cloudflare R2 (default), Google Cloud Storage (optional)

## Branches

`main` is the base worker. `RTX-4090`, `RTX-5090` and `AAE-6000` hold GPU-specific variants that have not been merged into `main`.

## Limitations

- Needs an NVIDIA GPU with enough VRAM for the selected LTX-Video model.
- No automated tests beyond the R2 wiring check in `test_r2_wiring.py`.

## License

MIT, see [LICENSE](LICENSE).

## Contact

Anubhav Gupta, [LinkedIn](https://www.linkedin.com/in/anubhav-gupta-92410a244/)
