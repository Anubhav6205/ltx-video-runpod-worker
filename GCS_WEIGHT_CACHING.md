# GCS Weight Caching for RunPod

This guide explains how to use Google Cloud Storage to cache model weights for faster RunPod pod startup times.

## TL;DR - Realistic Startup Times

| Method | First Run | Subsequent | Persistent? | Cost |
|--------|-----------|-----------|------------|------|
| **HuggingFace Only** | 15-20 min | 15-20 min | ❌ No | Free |
| **HuggingFace + Pod Persistent Volume** | 15-20 min | 30-60 sec | ✅ Yes | ~$0.07/GB/mo |
| **GCS + Local Cache** | 60-120 sec | 30-60 sec | ❌ No | ~$0.02/GB/mo storage + egress |
| **HuggingFace + Network Volume** | 15-20 min | 30-60 sec | ✅ Yes | ~$0.07/GB/mo |

## ⚠️ Reality Check: 30 Seconds is Not Achievable with 15GB Download

To download 15GB in 30 seconds requires **4.3 Gbps network speed** (theoretical). Real-world limits:

- HuggingFace: 50-200 Mbps (regional)
- GCS egress: 200-500 Mbps (regional)
- RunPod network: 500 Mbps - 1 Gbps

**Realistic achievable targets:**
- GCS with parallel downloads: **60-120 seconds** (~125-250 Mbps)
- RunPod Network Volume: **30-60 seconds** (local, 1+ Gbps)
- Pod local volume: **40-80 seconds** (local disk)

---

## Setup: GCS Weight Caching

### Step 1: Download Weights from HuggingFace (One-Time)

First, download weights from HuggingFace using the standard method:

```bash
# On any machine with GPU or RunPod pod
python3 -c "from app.bootstrap import init_model; init_model()"

# This downloads ~15-25 GB to /models/hf_hub
ls -lah /models/hf_hub/
```

### Step 2: Upload to GCS (One-Time Setup)

```bash
# Set your GCS bucket
export GCS_BUCKET=your-bucket-name
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Upload weights to GCS
python3 scripts/download_weights_gcs.py upload

# Expected time: 30-60 minutes (depending on upload bandwidth)
# Expected size in GCS: ~15-25 GB
```

### Step 3: Configure RunPod with GCS Caching

When deploying on RunPod, use the alternative entrypoint:

```yaml
Container Image: your-registry/ltx-video-worker:latest

Environment Variables:
  USE_GCS_CACHE: true                    # Enable GCS caching
  GCS_BUCKET: your-bucket-name          # Where weights are stored
  GOOGLE_APPLICATION_CREDENTIALS: /path/to/key.json

Volume Mounts:
  - /workspace (for outputs)
  # NO /models mount (will be populated from GCS)
```

### Step 4: Alternative Entrypoint (Optional)

If you want to use GCS caching:

```dockerfile
# In Dockerfile, use alternative entrypoint:
COPY entrypoint-gcs.sh /app/entrypoint-gcs.sh
RUN chmod +x /app/entrypoint-gcs.sh
ENTRYPOINT ["/app/entrypoint-gcs.sh"]
```

---

## How It Works

### Without GCS Caching (Standard)

```
Pod Startup:
1. Check /models → Empty (no persistent volume)
2. Download from HuggingFace → /models/hf_hub (15-20 minutes)
3. Load model into VRAM
4. Server ready

Pod Restart:
1. Repeat entire process (15-20 minutes every time!)
```

### With GCS Caching

```
Pod Startup (First):
1. Check /models → Empty
2. Download from GCS → /models (60-120 seconds)
3. Load model into VRAM
4. Server ready

Pod Startup (Subsequent):
1. Check /models → Files exist!
2. Load from local cache into VRAM (30-60 seconds)
3. Server ready
```

---

## Recommended Production Setup

### Option A: Network Volume (BEST - No GCS Needed)

```yaml
# RunPod Configuration
Container Disk: 20 GB
Network Volume: 50 GB → /models  # Persistent across pods

Environment Variables:
  PRELOAD_MODEL: true
```

**Advantages:**
- ✅ 30-60 second startup after first run
- ✅ Shared across multiple pods
- ✅ Fast local access (1+ Gbps)
- ✅ Works for all regions

**Disadvantages:**
- ❌ Need to wait 15-20 minutes on first run
- ❌ Network volume fee: ~$0.07/GB/month

---

### Option B: GCS Caching (For Ephemeral Pods)

```yaml
# RunPod Configuration
Container Disk: 20 GB
# No persistent volume needed

Environment Variables:
  USE_GCS_CACHE: true
  GCS_BUCKET: your-bucket
  GOOGLE_APPLICATION_CREDENTIALS: /workspace/gcs-key.json

Volume Mounts:
  - /workspace (for GCS credentials and outputs)
```

**Advantages:**
- ✅ Works with ephemeral/spot pods
- ✅ 60-120 second startup (3x faster than HF)
- ✅ Cheaper storage: ~$0.02/GB/month
- ✅ Easy to use multiple pods

**Disadvantages:**
- ❌ Slower than Network Volume (60-120 sec vs 30-60 sec)
- ❌ GCS egress costs: ~$0.01/GB
- ❌ Every pod startup needs download

**Cost Comparison (per pod per month, assuming 30 pods):**
- Network Volume: $3.50 (single shared volume)
- GCS + 30 pods: $0.60 storage + $4.50 egress = $5.10

---

## Performance Tuning

### To Maximize GCS Download Speed

```bash
# In entrypoint-gcs.sh, modify download to use parallel transfers:
export GRPC_PYTHON_BUILD_WITH_CYTHON=1
gsutil -m cp -r gs://bucket/ltx-models /models/

# Expected speedup: 2-3x faster
```

### To Optimize Network Volume

1. **Create volume in same region as pod**: Avoid cross-region latency
2. **Use SSD volume if available**: Faster than HDD
3. **Pre-load weights in volume**: During first deployment, don't interrupt

---

## Commands Reference

### Download Weights from GCS

```bash
# Check if weights cached locally
python3 scripts/download_weights_gcs.py check

# Download from GCS
python3 scripts/download_weights_gcs.py download

# If download fails, falls back to HuggingFace
```

### Upload Weights to GCS

```bash
# One-time setup (after downloading from HF)
export GCS_BUCKET=your-bucket
python3 scripts/download_weights_gcs.py upload

# Verify upload
gsutil ls -r gs://your-bucket/ltx-models/
```

### Check Download Speed

```bash
# Measure GCS download speed
time gsutil -m cp gs://your-bucket/ltx-models/model.safetensors ./

# Measure HuggingFace speed (from python)
python3 -c "
from huggingface_hub import hf_hub_download
import time
start = time.time()
hf_hub_download('Lightricks/LTX-Video', 'ltxv-2b-0.9.8-distilled.safetensors')
print(f'Downloaded in {time.time() - start:.1f}s')
"
```

---

## Troubleshooting

### GCS Download Fails

```bash
# Check GCS credentials
python3 -c "from google.cloud import storage; client = storage.Client(); print('✅ Auth OK')"

# Check bucket access
gsutil ls gs://your-bucket/ltx-models/

# Check network connectivity
ping gs://your-bucket  # Won't work but shows DNS resolution
curl -I https://storage.googleapis.com
```

### Weights Missing from GCS

```bash
# List what's in GCS
gsutil ls -r gs://your-bucket/ltx-models/

# Expected files:
# - model.safetensors
# - config.json
# - model_index.json
# - text_encoder/*
# - vae/*
```

### Downloads Slower Than Expected

```bash
# Check network speed
# From RunPod:
docker run --rm curlimages/curl sh -c "time curl -o /dev/null https://storage.googleapis.com/storage/v1/b"

# Check GCS region
gsutil bucketinfo gs://your-bucket | grep location

# Consider uploading to same region as RunPod
```

---

## My Recommendation

**For your use case:**

1. **Start with Network Volume** (easiest, fastest)
   - 30-60 second startup
   - Shared across pods
   - Set it and forget it

2. **If you need lower storage costs**, use GCS
   - 60-120 second startup
   - Pay only for what you use
   - Good for spot instances

3. **Don't use HuggingFace-only** unless free/testing
   - 15-20 minutes every startup
   - Wastes bandwidth and money
   - Terrible user experience

**Best bang for buck:** RunPod Network Volume - $3.50/month for unlimited fast access to weights!
