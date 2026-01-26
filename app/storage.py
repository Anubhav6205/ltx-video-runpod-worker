"""
Storage integration for uploading generated videos.

Default: Cloudflare R2 (S3-compatible).
Fallback/optional: Google Cloud Storage (GCS).
"""

from pathlib import Path
from datetime import timedelta

import boto3
from botocore.config import Config

from google.cloud import storage
from google.oauth2 import service_account
from google.cloud.exceptions import GoogleCloudError

from app.settings import settings
from app.logging_config import get_logger
from app.utils import format_bytes

logger = get_logger(__name__)


def _normalize_public_base(public_base: str, bucket: str) -> str:
    try:
        public_base = public_base.format(bucket=bucket)
    except Exception:
        # If the user provided a literal `{...}` that isn't a format placeholder, keep it as-is.
        pass
    return public_base.rstrip("/")


def _is_r2_configured() -> bool:
    return bool(
        settings.r2_bucket
        and (settings.r2_endpoint_url or settings.r2_account_id)
        and settings.r2_access_key_id
        and settings.r2_secret_access_key
    )


def _is_gcs_configured() -> bool:
    return bool(settings.gcs_bucket)


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


def get_r2_client():
    """
    Create and return an R2 (S3-compatible) client configured from settings.

    Returns:
        boto3 S3 client

    Raises:
        ValueError: If R2 configuration is missing or invalid
    """
    if not settings.r2_bucket:
        raise ValueError("R2 bucket not configured (R2_BUCKET)")

    endpoint_url = settings.r2_endpoint_url
    if not endpoint_url and settings.r2_account_id:
        endpoint_url = f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"

    if not endpoint_url:
        raise ValueError("R2 endpoint not configured (R2_ENDPOINT_URL or R2_ACCOUNT_ID)")

    if not settings.r2_access_key_id or not settings.r2_secret_access_key:
        raise ValueError("R2 credentials not configured (R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)")

    try:
        # R2 works best with path-style addressing for custom endpoints.
        config = Config(signature_version="s3v4", s3={"addressing_style": "path"})
        session = boto3.session.Session(
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region or "auto",
        )
        return session.client("s3", endpoint_url=endpoint_url, config=config)

    except Exception as e:
        logger.error(f"Failed to create R2 client: {str(e)}")
        raise ValueError(f"Failed to create R2 client: {str(e)}")


def upload_file_and_get_url_gcs(local_path: Path, key: str) -> str:
    """
    Upload a file to GCS and return a public/signed URL.

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


def upload_file_and_get_url_r2(local_path: Path, key: str) -> str:
    """
    Upload a file to Cloudflare R2 and return a public/signed URL.

    If `R2_PUBLIC_BASE_URL` (or `R2_PUBLIC_BASE`) is set, returns `${R2_PUBLIC_BASE_URL}/{key}`
    (supports `{bucket}` placeholder). Otherwise returns a presigned URL (TTL controlled by
    `R2_PRESIGN_TTL_S` or `R2_SIGNED_URL_TTL_S`).
    """
    if not local_path.exists():
        raise ValueError(f"File not found: {local_path}")

    try:
        logger.info(f"Uploading {local_path.name} to R2...")
        logger.info(f"Bucket: {settings.r2_bucket}, Key: {key}")

        client = get_r2_client()

        file_size = local_path.stat().st_size
        logger.info(f"Uploading {format_bytes(file_size)}...")

        client.upload_file(
            str(local_path),
            settings.r2_bucket,
            key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        logger.info("Upload complete")

        if settings.r2_public_base:
            base = _normalize_public_base(settings.r2_public_base, settings.r2_bucket)
            url = f"{base}/{key}"
            logger.info(f"Using public URL: {url}")
            return url

        logger.info(f"Generating signed URL (TTL: {settings.r2_signed_url_ttl_s}s)")
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket, "Key": key},
            ExpiresIn=settings.r2_signed_url_ttl_s,
        )

    except Exception as e:
        logger.error(f"R2 upload failed: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to upload to R2: {str(e)}")


def upload_file_and_get_url(local_path: Path, key: str) -> str:
    backend = (settings.storage_backend or "r2").strip().lower()

    # Make R2 the default, but keep legacy deployments working when R2 isn't configured.
    if backend == "r2":
        if _is_r2_configured():
            return upload_file_and_get_url_r2(local_path, key)
        if _is_gcs_configured():
            logger.warning("R2 selected but not configured; falling back to GCS")
            return upload_file_and_get_url_gcs(local_path, key)
        raise ValueError("R2 selected but not configured (set R2_* env vars)")

    if backend == "gcs":
        if _is_gcs_configured():
            return upload_file_and_get_url_gcs(local_path, key)
        if _is_r2_configured():
            logger.warning("GCS selected but not configured; falling back to R2")
            return upload_file_and_get_url_r2(local_path, key)
        raise ValueError("GCS selected but not configured (set GCS_* env vars)")

    raise ValueError("Invalid STORAGE_BACKEND (must be 'r2' or 'gcs')")


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
    prefix = (settings.storage_key_prefix or "clips").strip().strip("/")
    key = f"{prefix}/{job_id}.mp4" if prefix else f"{job_id}.mp4"
    return upload_file_and_get_url(local_path, key)
