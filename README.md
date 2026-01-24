# LTX-Video RunPod Worker

A production-ready FastAPI service for generating videos from images using **LTX-Video 2B distilled model**. Designed for deployment on RunPod GPU pods (RTX 4090) with Google Cloud Storage (GCS) integration.

## Features

- **Image-to-Video Generation** using LTX-Video 2B distilled model
- **REST API** with FastAPI (OpenAPI docs at `/docs`)
- **GCS Integration** for video storage and public URL generation
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
  "clip_url": "https://storage.googleapis.com/my-bucket/clips/f47ac10b-58cc-4372-a567-0e02b2c3d479.mp4",
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
- Google Cloud Storage bucket and credentials

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

4. **Configure Google Cloud credentials**

Create a service account in Google Cloud Console with Storage Object Admin permissions, download the JSON key file, and set up authentication:

```bash
# Option 1: Set environment variable (recommended for local dev)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"

# Option 2: Specify in .env file
# GCS_SERVICE_ACCOUNT_KEY=/path/to/your-service-account-key.json
```

5. **Configure environment**

```bash
cp .env.example .env
# Edit .env and fill in your configuration:
# - GCS_BUCKET=your-bucket-name
# - GCS_PROJECT_ID=your-project-id
# - GCS_SERVICE_ACCOUNT_KEY=/path/to/key.json (if not using GOOGLE_APPLICATION_CREDENTIALS)
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
  -v /path/to/your-service-account-key.json:/app/gcs-key.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json \
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
2. **Configure GCS storage** for video uploads
3. **Prepare service account key** as a base64 string (for environment variable)

```bash
# Convert service account key to base64
base64 -w 0 your-service-account-key.json > key-base64.txt
```

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
     GCS_BUCKET=your-bucket-name
     GCS_PROJECT_ID=your-project-id
     GCS_SERVICE_ACCOUNT_KEY=/tmp/gcs-key.json
     HF_TOKEN=your-hf-token (optional)
     WARMUP_ON_START=true (optional)
     ```
   - **Secret Files**: Upload your service account JSON key file to `/tmp/gcs-key.json`
     - Or use base64 encoded key in environment variable and decode in entrypoint

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
- **GCS Credentials**: Store service account key securely - consider using RunPod secrets or environment variables

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
| `GCS_BUCKET` | *required* | GCS bucket name |
| `GCS_PROJECT_ID` | *optional* | GCS project ID |
| `GCS_SERVICE_ACCOUNT_KEY` | *optional* | Path to service account JSON key |
| `GCS_PUBLIC_BASE` | *optional* | Public CDN URL (if bucket is public) |
| `GCS_MAKE_PUBLIC` | `false` | Auto-make uploaded blobs public |
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

### Google Cloud Storage Setup

1. **Create a GCS bucket**

```bash
gsutil mb -p your-project-id -c STANDARD -l us-central1 gs://your-bucket-name
```

2. **Create a service account**

```bash
gcloud iam service-accounts create ltx-worker \
    --display-name="LTX Video Worker"
```

3. **Grant permissions**

```bash
gsutil iam ch serviceAccount:ltx-worker@your-project-id.iam.gserviceaccount.com:objectAdmin \
    gs://your-bucket-name
```

4. **Create and download key**

```bash
gcloud iam service-accounts keys create ~/ltx-worker-key.json \
    --iam-account=ltx-worker@your-project-id.iam.gserviceaccount.com
```

5. **Configure .env**

```bash
GCS_BUCKET=your-bucket-name
GCS_PROJECT_ID=your-project-id
GCS_SERVICE_ACCOUNT_KEY=/path/to/ltx-worker-key.json
```

### URL Generation Options

The worker supports three methods for generating video URLs:

#### 1. Public Base URL (CDN)

```bash
# .env configuration for CDN
GCS_BUCKET=your-bucket-name
GCS_PUBLIC_BASE=https://cdn.example.com
```

Returns: `https://cdn.example.com/clips/{job_id}.mp4`

**Use when**: You have a CDN or Cloud Load Balancer in front of your GCS bucket.

#### 2. Make Blobs Public

```bash
# .env configuration for public bucket
GCS_BUCKET=your-bucket-name
GCS_MAKE_PUBLIC=true
```

Returns: `https://storage.googleapis.com/your-bucket-name/clips/{job_id}.mp4`

**Use when**: Your bucket is configured for public access. Blobs are automatically made public on upload.

#### 3. Signed URLs (Default)

```bash
# .env configuration for private bucket
GCS_BUCKET=your-bucket-name
GCS_SIGNED_URL_TTL_S=3600
```

Returns: `https://storage.googleapis.com/your-bucket-name/clips/{job_id}.mp4?X-Goog-Signature=...`

**Use when**: You want temporary access to private files. URLs expire after TTL seconds (default 1 hour).

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

### GCS Upload Failures

**Symptom**: Generation succeeds but upload fails

**Solutions**:
- Verify GCS credentials are correctly configured
- Check service account has Storage Object Admin role
- Verify bucket exists: `gsutil ls gs://your-bucket-name`
- Test bucket write permissions: `gsutil cp test.txt gs://your-bucket-name/`
- Check GOOGLE_APPLICATION_CREDENTIALS environment variable
- Ensure network connectivity to storage.googleapis.com

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
- **Storage**: Google Cloud Storage
#   M e d i a - W o r k e r  
 