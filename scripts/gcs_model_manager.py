#!/usr/bin/env python3
"""
GCS Model Manager - Download/Upload LTX-Video weights from/to Google Cloud Storage
Faster than HuggingFace, works seamlessly across RunPod pods
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import json

from google.cloud import storage
from google.oauth2 import service_account

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCSModelManager:
    """Manage LTX-Video model weights in Google Cloud Storage"""

    def __init__(self, gcs_bucket: str, gcs_project_id: Optional[str] = None):
        """
        Initialize GCS client

        Args:
            gcs_bucket: GCS bucket name (e.g., "my-models")
            gcs_project_id: GCP project ID (optional, uses default if not provided)
        """
        self.gcs_bucket = gcs_bucket
        self.gcs_project_id = gcs_project_id
        self.model_prefix = "ltx-video-2b-distilled"  # GCS folder prefix

        try:
            self.client = storage.Client(project=gcs_project_id)
            self.bucket = self.client.bucket(gcs_bucket)
            logger.info(f"✅ Connected to GCS bucket: gs://{gcs_bucket}")
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            raise

    def upload_weights(self, local_cache_dir: str) -> bool:
        """
        Upload model weights from local HuggingFace cache to GCS
        Run this ONCE to prepare weights for pod distribution

        Args:
            local_cache_dir: Path to HuggingFace cache (e.g., /models/hf_hub)

        Returns:
            bool: Success status
        """
        logger.info("=" * 60)
        logger.info("Uploading LTX-Video weights to GCS...")
        logger.info("=" * 60)

        cache_path = Path(local_cache_dir)

        # Find the model directory
        model_dir = cache_path / "models--Lightricks--LTX-Video"

        if not model_dir.exists():
            logger.error(f"Model cache not found at {model_dir}")
            logger.error("First download weights using: python3 -c \"from app.bootstrap import init_model; init_model()\"")
            return False

        # Find the snapshot (usually a hash)
        snapshots = list(model_dir.glob("snapshots/*"))
        if not snapshots:
            logger.error("No model snapshots found")
            return False

        model_snapshot = snapshots[0]
        logger.info(f"Found model snapshot at: {model_snapshot}")

        # Upload files
        try:
            uploaded_count = 0
            total_size = 0

            file_list = list(model_snapshot.glob("**/*"))
            logger.info(f"Found {len([f for f in file_list if f.is_file()])} files to upload")

            for file_path in file_list:
                if not file_path.is_file():
                    continue

                relative_path = file_path.relative_to(model_snapshot)
                gcs_path = f"{self.model_prefix}/{relative_path}"

                file_size = file_path.stat().st_size
                total_size += file_size

                logger.info(f"Uploading {file_path.name} ({file_size/1024/1024:.1f} MB)...")

                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(file_path)
                uploaded_count += 1

            logger.info("=" * 60)
            logger.info(f"✅ Successfully uploaded {uploaded_count} files")
            logger.info(f"Total size: {total_size/1024/1024/1024:.1f} GB")
            logger.info(f"GCS path: gs://{self.gcs_bucket}/{self.model_prefix}/")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_weights(self, output_cache_dir: str, parallel: bool = True) -> bool:
        """
        Download model weights from GCS to local cache
        Run on pod startup to populate /models/hf_hub

        Args:
            output_cache_dir: Path where to save weights (e.g., /models/hf_hub)
            parallel: Use parallel downloads (faster)

        Returns:
            bool: Success status
        """
        logger.info("=" * 60)
        logger.info("Downloading LTX-Video weights from GCS...")
        logger.info("=" * 60)

        output_path = Path(output_cache_dir)

        # Create target structure
        model_dir = output_path / "models--Lightricks--LTX-Video" / "snapshots" / "main"
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Target directory: {model_dir}")

        try:
            # List all blobs
            blobs = list(self.bucket.list_blobs(prefix=self.model_prefix))

            if not blobs:
                logger.error(f"No files found in GCS under {self.model_prefix}")
                logger.error("Did you run upload_weights() first?")
                return False

            logger.info(f"Found {len(blobs)} files to download")

            # Download files
            downloaded_count = 0
            total_size = 0

            for blob in blobs:
                if blob.name.endswith('/'):
                    continue  # Skip directory markers

                # Extract relative path
                relative_path = blob.name[len(self.model_prefix)+1:]
                local_path = model_dir / relative_path

                # Create parent directories
                local_path.parent.mkdir(parents=True, exist_ok=True)

                blob_size = blob.size if hasattr(blob, 'size') else 0
                total_size += blob_size

                logger.info(f"Downloading {relative_path} ({blob_size/1024/1024:.1f} MB)...")
                blob.download_to_filename(str(local_path))
                downloaded_count += 1

            logger.info("=" * 60)
            logger.info(f"✅ Successfully downloaded {downloaded_count} files")
            logger.info(f"Total size: {total_size/1024/1024/1024:.1f} GB")
            logger.info(f"Saved to: {model_dir}")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def check_local_weights(self, cache_dir: str) -> bool:
        """
        Check if weights are cached locally

        Args:
            cache_dir: Path to cache directory

        Returns:
            bool: True if weights exist
        """
        model_dir = Path(cache_dir) / "models--Lightricks--LTX-Video"
        exists = model_dir.exists()

        if exists:
            files = list(model_dir.glob("**/*"))
            file_count = len([f for f in files if f.is_file()])
            logger.info(f"✅ Found {file_count} cached weight files")
        else:
            logger.info("❌ No cached weights found")

        return exists

    def list_gcs_weights(self) -> bool:
        """
        List all weight files in GCS
        """
        try:
            blobs = list(self.bucket.list_blobs(prefix=self.model_prefix))

            if not blobs:
                logger.info(f"No files in GCS under {self.model_prefix}")
                return False

            logger.info(f"Files in gs://{self.gcs_bucket}/{self.model_prefix}:")
            total_size = 0

            for blob in blobs:
                if not blob.name.endswith('/'):
                    size_mb = blob.size / 1024 / 1024
                    total_size += blob.size
                    print(f"  {blob.name} ({size_mb:.1f} MB)")

            logger.info(f"Total: {total_size/1024/1024/1024:.1f} GB")
            return True

        except Exception as e:
            logger.error(f"Failed to list GCS weights: {e}")
            return False


def main():
    """CLI interface"""

    if len(sys.argv) < 2:
        print("Usage: python3 gcs_model_manager.py [upload|download|check|list]")
        print("\nCommands:")
        print("  upload <cache_dir>   - Upload weights from local HuggingFace cache to GCS")
        print("  download <output_dir> - Download weights from GCS to local cache")
        print("  check <cache_dir>    - Check if weights are cached locally")
        print("  list                 - List all weights in GCS")
        print("\nEnvironment Variables:")
        print("  GCS_BUCKET          - GCS bucket name (REQUIRED)")
        print("  GCP_PROJECT_ID      - GCP project ID (optional)")
        sys.exit(1)

    # Get GCS bucket from environment
    gcs_bucket = os.getenv("GCS_BUCKET")
    if not gcs_bucket:
        logger.error("GCS_BUCKET environment variable not set")
        sys.exit(1)

    gcp_project_id = os.getenv("GCP_PROJECT_ID")

    try:
        manager = GCSModelManager(gcs_bucket, gcp_project_id)
    except Exception as e:
        logger.error(f"Failed to initialize GCS manager: {e}")
        sys.exit(1)

    command = sys.argv[1]

    if command == "upload":
        if len(sys.argv) < 3:
            logger.error("Usage: python3 gcs_model_manager.py upload <cache_dir>")
            sys.exit(1)
        cache_dir = sys.argv[2]
        success = manager.upload_weights(cache_dir)
        sys.exit(0 if success else 1)

    elif command == "download":
        if len(sys.argv) < 3:
            logger.error("Usage: python3 gcs_model_manager.py download <output_dir>")
            sys.exit(1)
        output_dir = sys.argv[2]
        success = manager.download_weights(output_dir)
        sys.exit(0 if success else 1)

    elif command == "check":
        if len(sys.argv) < 3:
            logger.error("Usage: python3 gcs_model_manager.py check <cache_dir>")
            sys.exit(1)
        cache_dir = sys.argv[2]
        exists = manager.check_local_weights(cache_dir)
        sys.exit(0 if exists else 1)

    elif command == "list":
        success = manager.list_gcs_weights()
        sys.exit(0 if success else 1)

    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
