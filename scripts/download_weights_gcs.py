#!/usr/bin/env python3
"""
Download LTX-Video model weights from GCS instead of HuggingFace
Faster than downloading from HF, useful for ephemeral RunPod pods
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging

from google.cloud import storage
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model information
MODEL_FILES = {
    "ltxv-2b-0.9.8-distilled.safetensors": "transformer weights",
    "config.json": "config",
    "model_index.json": "model index",
}

GCS_MODEL_PREFIX = "ltx-models"  # GCS folder prefix


def upload_weights_to_gcs():
    """
    Upload model weights from HuggingFace cache to GCS (one-time setup)
    Run this once to cache weights in GCS
    """
    logger.info("Uploading LTX-Video weights to GCS...")

    if not settings.gcs_bucket:
        logger.error("GCS_BUCKET not configured!")
        return False

    try:
        client = storage.Client()
        bucket = client.bucket(settings.gcs_bucket)

        hf_cache = Path(settings.hf_hub_cache) / "models--Lightricks--LTX-Video" / "snapshots"

        if not hf_cache.exists():
            logger.error(f"HF cache not found at {hf_cache}. Download model first using HuggingFace!")
            return False

        # Find the snapshot directory (usually named with a hash)
        snapshots = list(hf_cache.glob("*"))
        if not snapshots:
            logger.error("No model snapshots found in HF cache")
            return False

        model_dir = snapshots[0]
        logger.info(f"Found model at {model_dir}")

        uploaded_count = 0
        for file_path in model_dir.glob("**/*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_dir)
                gcs_path = f"{GCS_MODEL_PREFIX}/{relative_path}"

                logger.info(f"Uploading {file_path.name} → gs://{settings.gcs_bucket}/{gcs_path}")

                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(file_path)
                uploaded_count += 1

        logger.info(f"✅ Uploaded {uploaded_count} files to GCS!")
        return True

    except Exception as e:
        logger.error(f"Failed to upload weights: {e}")
        return False


def download_weights_from_gcs() -> bool:
    """
    Download LTX-Video weights from GCS cache
    Faster than downloading from HuggingFace for RunPod pods
    """
    logger.info("Downloading LTX-Video weights from GCS...")

    if not settings.gcs_bucket:
        logger.error("GCS_BUCKET not configured!")
        return False

    try:
        client = storage.Client()
        bucket = client.bucket(settings.gcs_bucket)

        # Create cache directory
        cache_dir = Path(settings.hf_hub_cache) / "models--Lightricks--LTX-Video" / "snapshots" / "main"
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading to {cache_dir}")

        # List all blobs under the model prefix
        blobs = bucket.list_blobs(prefix=GCS_MODEL_PREFIX)

        downloaded_count = 0
        for blob in blobs:
            if blob.name.endswith('/'):
                continue  # Skip directories

            # Extract relative path (remove prefix)
            relative_path = blob.name[len(GCS_MODEL_PREFIX)+1:]
            local_path = cache_dir / relative_path

            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {blob.name} → {local_path}")
            blob.download_to_filename(local_path)
            downloaded_count += 1

        if downloaded_count == 0:
            logger.warning(f"No files found in GCS under {GCS_MODEL_PREFIX}")
            logger.warning("Did you run upload_weights_to_gcs() first?")
            return False

        logger.info(f"✅ Downloaded {downloaded_count} files from GCS!")
        return True

    except Exception as e:
        logger.error(f"Failed to download weights from GCS: {e}")
        return False


def check_weights_cached() -> bool:
    """Check if weights are already cached locally"""
    cache_dir = Path(settings.hf_hub_cache) / "models--Lightricks--LTX-Video"
    return cache_dir.exists()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 download_weights_gcs.py [download|upload|check]")
        print("\nCommands:")
        print("  download  - Download weights from GCS to local cache")
        print("  upload    - Upload weights from HF cache to GCS (one-time setup)")
        print("  check     - Check if weights are cached locally")
        sys.exit(1)

    command = sys.argv[1]

    if command == "download":
        success = download_weights_from_gcs()
        sys.exit(0 if success else 1)
    elif command == "upload":
        success = upload_weights_to_gcs()
        sys.exit(0 if success else 1)
    elif command == "check":
        if check_weights_cached():
            logger.info("✅ Weights are cached locally")
            sys.exit(0)
        else:
            logger.info("❌ Weights are NOT cached locally")
            sys.exit(1)
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)
