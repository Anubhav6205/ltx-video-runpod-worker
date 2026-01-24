"""
Google Cloud Storage (GCS) integration for uploading generated videos
Supports both public buckets and signed URLs
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import timedelta

from google.cloud import storage
from google.oauth2 import service_account
from google.cloud.exceptions import GoogleCloudError

from app.settings import settings
from app.logging_config import get_logger
from app.utils import format_bytes

logger = get_logger(__name__)


def get_gcs_client():
    """
    Create and return a GCS client configured from settings

    Returns:
        google.cloud.storage.Client

    Raises:
        ValueError: If GCS credentials are missing or invalid
    """
    if not settings.gcs_bucket:
        raise ValueError("GCS bucket not configured (GCS_BUCKET)")

    try:
        # If service account key path is provided, use it
        if settings.gcs_service_account_key:
            credentials = service_account.Credentials.from_service_account_file(
                settings.gcs_service_account_key
            )
            client = storage.Client(
                credentials=credentials,
                project=settings.gcs_project_id
            )
        else:
            # Use application default credentials (from GOOGLE_APPLICATION_CREDENTIALS env var)
            client = storage.Client(project=settings.gcs_project_id)

        return client

    except Exception as e:
        logger.error(f"Failed to create GCS client: {str(e)}")
        raise ValueError(f"Failed to create GCS client: {str(e)}")


def upload_file_and_get_url(local_path: Path, key: str) -> str:
    """
    Upload a file to GCS and return a public URL

    Args:
        local_path: Path to local file
        key: GCS object key (path within bucket)

    Returns:
        Public URL to access the uploaded file

    Raises:
        ValueError: If upload fails or configuration is invalid
    """
    if not local_path.exists():
        raise ValueError(f"File not found: {local_path}")

    try:
        logger.info(f"Uploading {local_path.name} to GCS...")
        logger.info(f"Bucket: {settings.gcs_bucket}, Key: {key}")

        # Get GCS client
        client = get_gcs_client()
        bucket = client.bucket(settings.gcs_bucket)
        blob = bucket.blob(key)

        # Upload file
        file_size = local_path.stat().st_size
        logger.info(f"Uploading {format_bytes(file_size)}...")

        blob.upload_from_filename(
            str(local_path),
            content_type="video/mp4"
        )

        logger.info("Upload complete")

        # Generate URL
        if settings.gcs_public_base:
            # Use public base URL (for CDN or public buckets)
            url = f"{settings.gcs_public_base.rstrip('/')}/{key}"
            logger.info(f"Using public URL: {url}")
        elif settings.gcs_make_public:
            # Make the blob publicly accessible and return public URL
            blob.make_public()
            url = blob.public_url
            logger.info(f"Made blob public: {url}")
        else:
            # Generate signed URL
            logger.info(f"Generating signed URL (TTL: {settings.gcs_signed_url_ttl_s}s)")
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=settings.gcs_signed_url_ttl_s),
                method="GET"
            )

        return url

    except GoogleCloudError as e:
        logger.error(f"GCS upload failed: {str(e)}")
        raise ValueError(f"Failed to upload to GCS: {str(e)}")

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to upload file: {str(e)}")


def upload_video(job_id: str, local_path: Path) -> str:
    """
    Upload a generated video and return its public URL

    Args:
        job_id: Unique job identifier
        local_path: Path to local MP4 file

    Returns:
        Public URL to access the video
    """
    # Use job_id as filename to avoid collisions
    key = f"clips/{job_id}.mp4"
    return upload_file_and_get_url(local_path, key)
