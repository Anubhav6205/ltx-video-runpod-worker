# Fast Startup Guide: Eliminate 20-Minute Weight Download

This guide shows you how to achieve **30-60 second pod startup** by pre-caching the 25GB model weights.

## 🎯 Goal: Pod starts in 30-60 seconds (not 20 minutes)

---

## ⚡ Option 1: Bake Weights into Docker Image (FASTEST STARTUP)

### How It Works
Download model weights **during Docker build**, include them in the image itself.

**Result:**
- ✅ Pod startup: **30-60 seconds** (immediate!)
- ✅ No persistent volume needed
- ✅ Perfect for spot instances
- ✅ Consistent across all pods

**Trade-offs:**
- ❌ Docker image: **~37 GB** (huge!)
- ❌ Build time: **+20 minutes** (one-time)
- ❌ Push to registry: **+30 minutes** (one-time)
- ❌ Registry storage: **~$15-20/month**
- ❌ Model updates require full rebuild

### Build Instructions

```bash
# Build image with weights baked in
docker build -f Dockerfile.with-weights -t ltx-video-worker:with-weights .

# Expected output:
# [+] Building 2100.5s (15/15) FINISHED
# ...
# Downloading LTX-Video weights into Docker image...
# This will take 15-20 minutes but only happens once during build
# ...
# Weights successfully baked into image!
# 25G    /models/hf_hub

# Image size check
docker images ltx-video-worker:with-weights
# REPOSITORY              TAG           SIZE
# ltx-video-worker        with-weights  37.2GB

# Tag and push to registry
docker tag ltx-video-worker:with-weights your-registry/ltx-video-worker:with-weights
docker push your-registry/ltx-video-worker:with-weights  # Takes 20-30 minutes
```

### Deploy on RunPod

```yaml
Container Image: your-registry/ltx-video-worker:with-weights

Storage:
  Container Disk: 40 GB  # Larger for the 37GB image
  # NO Volume Disk needed!

Environment Variables:
  PORT: 8000
  PRELOAD_MODEL: false  # Already loaded in image!

Startup Time: 30-60 seconds ⚡
```

---

## 🔄 Option 2: RunPod Network Volume with Pre-loaded Weights

### How It Works
Create a Network Volume once, download weights to it, reuse across all pods.

**Result:**
- ✅ Pod startup: **30-60 seconds** (after volume created)
- ✅ Share volume across multiple pods
- ✅ Docker image stays small (12GB)
- ✅ Easy to update model (just update volume)

**Trade-offs:**
- ❌ First-time setup: **20 minutes** (creating volume + downloading weights)
- ❌ Volume cost: **$3.50/month** (50GB)
- ❌ Requires persistent infrastructure

### Setup Instructions

#### Step 1: Create Network Volume on RunPod

1. Go to RunPod Console → **Storage** → **Network Volumes**
2. Click **Create Network Volume**
3. Configure:
   - **Name**: `ltx-models-cache`
   - **Size**: 50 GB
   - **Region**: Same as your pods (e.g., US-CA-1)
4. Click **Create**

#### Step 2: Populate Volume with Weights (One-Time)

Deploy a temporary pod to download weights:

```yaml
Container Image: your-registry/ltx-video-worker:latest
GPU: Any (RTX 3090, 4090, etc.)

Storage:
  Container Disk: 20 GB
  Network Volume: ltx-models-cache → /models

Environment Variables:
  PRELOAD_MODEL: true

# Let it run once to download weights (~20 minutes)
# Then stop the pod
```

**Verify weights are downloaded:**
```bash
# SSH into pod or check logs
ls -lah /models/hf_hub/
# Should see ~25GB of files
```

#### Step 3: Deploy Production Pods with Pre-loaded Volume

```yaml
Container Image: your-registry/ltx-video-worker:latest

Storage:
  Container Disk: 20 GB
  Network Volume: ltx-models-cache → /models  # Reuse same volume!

Environment Variables:
  PRELOAD_MODEL: true  # Will find cached weights

Startup Time: 30-60 seconds ⚡
```

**All pods using this volume will start in 30-60 seconds!**

---

## 🏆 Comparison: Which to Choose?

| Criteria | Baked Image | Network Volume | HF Only |
|----------|-------------|----------------|---------|
| **Startup Time** | 30-60 sec | 30-60 sec | 15-20 min |
| **First Run** | 30-60 sec | 20 min | 15-20 min |
| **Setup Complexity** | Medium | Low | None |
| **Docker Image Size** | 37 GB | 12 GB | 12 GB |
| **Monthly Cost** | $15-20 | $3.50 | Free |
| **Best For** | Spot instances | Production | Testing |
| **Model Updates** | Rebuild image | Update volume | Automatic |

---

## 💡 My Recommendation

### For Production (Best Value)
**Use Network Volume** (Option 2):
- Only $3.50/month
- 30-60 second startup
- Easy to manage
- Share across pods

### For Spot Instances
**Use Baked Image** (Option 1):
- Instant startup every time
- No volume dependencies
- Pay once for registry storage

### For Multi-Region Deployment
Use **one Network Volume per region** (or the baked image) to keep startup fast without re-downloading weights.

---

## 🚀 Quick Start: Network Volume (Recommended)

```bash
# 1. Build standard image (12GB)
docker build -t ltx-video-worker:latest .
docker push your-registry/ltx-video-worker:latest

# 2. Create RunPod Network Volume (via UI)
# Name: ltx-models-cache
# Size: 50 GB

# 3. Deploy temp pod to populate volume
# - Attach ltx-models-cache to /models
# - Set PRELOAD_MODEL=true
# - Wait 20 minutes
# - Stop pod

# 4. Deploy production pods
# - Attach ltx-models-cache to /models
# - Startup time: 30-60 seconds!
```

---

## 🔍 Verify Startup Time

After deployment, check logs to confirm fast startup:

```bash
# Expected fast startup logs:
================================================
Starting model initialization...
This may take several minutes on first run (downloading weights)
Pre-loading model weights...
Downloading transformer weights...
Transformer weights cached at: /models/hf_hub/...  ← Found in cache!
Loading LTX-Video pipeline...
[30 seconds later]
Starting FastAPI server...
================================================

# Total startup: 30-60 seconds ✅
```

---

## ⚠️ Important Notes

### For Baked Image Approach:
- Registry must support large images (Docker Hub: 10GB limit, use GitHub Container Registry or AWS ECR)
- Push time scales with image size (~1GB per minute)
- Consider image compression: `docker build --compress`

### For Network Volume Approach:
- Don't delete the volume accidentally!
- Back up volume occasionally (RunPod snapshots)
- Monitor volume usage: `du -sh /models`

### For All Approaches:
- Verify GPU is available before loading model
- Monitor VRAM usage: 18-22 GB for LTX-Video 2B
- Set appropriate health check timeout (60s+)

---

## 🎯 Summary

**Your goal: 30-60 second pod startup**

**Fastest solution:** Bake weights into Docker image (Option 1)
- Build: `docker build -f Dockerfile.with-weights`
- Deploy: Use 40GB container disk
- Result: Instant startup!

**Best value:** Network Volume (Option 2)
- Create volume once
- Download weights to it
- Reuse forever
- Only $3.50/month

**Both achieve your goal of eliminating the 20-minute download!** 🚀
