"""
Application settings loaded from environment variables and .env file
Uses pydantic-settings for validation and type safety
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration with defaults optimized for RunPod GPU pods
    All settings can be overridden via environment variables or .env file
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ===== Server Configuration =====
    port: int = Field(default=8000, description="HTTP server port")
    log_level: str = Field(default="INFO", description="Logging level")
    workdir: Path = Field(default=Path("/workspace"), description="Working directory")
    output_dir: Path = Field(default=Path("/workspace/outputs"), description="Video output directory")

    # ===== Concurrency & Timeouts =====
    max_concurrent_jobs: int = Field(default=1, description="Max simultaneous generations (1 = safe for single GPU)")
    image_download_timeout_s: int = Field(default=30, description="Timeout for downloading input images")
    generate_timeout_s: int = Field(default=300, description="Timeout for video generation")

    # ===== Hugging Face Configuration =====
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token (optional for public models)")
    hf_home: Path = Field(default=Path("/workspace/models/hf_home"), description="HuggingFace home directory")
    hf_hub_cache: Path = Field(default=Path("/workspace/models/hf_hub"), description="HuggingFace Hub cache directory")
    transformers_cache: Path = Field(default=Path("/workspace/models/hf_transformers"), description="Transformers cache directory")
    hf_hub_enable_hf_transfer: str = Field(default="1", description="Enable faster HF transfers")

    # ===== LTX-Video Model Configuration =====
    ltx_repo_id: str = Field(default="Lightricks/LTX-Video", description="HuggingFace repository ID")
    ltx_transformer_filename: str = Field(
        default="ltxv-13b-0.9.8-distilled-fp8.safetensors",
        description="Transformer weights filename"
    )
    torch_dtype: str = Field(default="bfloat16", description="Torch dtype (bfloat16, float16, float32)")
    use_fp8: bool = Field(default=True, description="Enable FP8 (float8_e4m3fn) loading for transformer")

    # ===== Default Generation Parameters =====
    default_width: int = Field(default=704, description="Default video width")
    default_height: int = Field(default=512, description="Default video height")
    default_fps: int = Field(default=24, description="Default frames per second")
    default_steps: int = Field(default=8, description="Default inference steps (distilled model optimized for 4-8)")
    default_guidance_scale: float = Field(default=1.0, description="Default guidance scale (1.0 for distilled)")
    default_negative_prompt: str = Field(
        default="worst quality, blurry, jittery, distorted, watermark, text",
        description="Default negative prompt"
    )

    # ===== Storage Configuration =====
    storage_backend: str = Field(
        default="r2",
        description="Storage backend for uploads: r2 or gcs"
    )
    storage_key_prefix: str = Field(
        default="clips",
        description="Prefix/folder for uploaded videos within the bucket"
    )

    # --- Cloudflare R2 (S3-compatible) ---
    r2_bucket: str = Field(default="", description="Cloudflare R2 bucket name")
    r2_account_id: Optional[str] = Field(default=None, description="Cloudflare account ID (used to build endpoint URL)")
    r2_endpoint_url: Optional[str] = Field(
        default=None,
        description="R2 S3 endpoint URL (overrides account-id based URL)"
    )
    r2_access_key_id: Optional[str] = Field(default=None, description="R2 access key ID")
    r2_secret_access_key: Optional[str] = Field(default=None, description="R2 secret access key")
    r2_region: str = Field(default="auto", description="S3 region (use 'auto' for R2)")
    r2_public_base: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_PUBLIC_BASE", "R2_PUBLIC_BASE_URL"),
        description="Public base URL for uploaded files (supports {bucket} placeholder)"
    )
    r2_signed_url_ttl_s: int = Field(
        default=3600,
        validation_alias=AliasChoices("R2_SIGNED_URL_TTL_S", "R2_PRESIGN_TTL_S"),
        description="Signed URL TTL in seconds"
    )

    # --- Google Cloud Storage (legacy/optional) ---
    gcs_bucket: str = Field(default="", description="GCS bucket name")
    gcs_project_id: Optional[str] = Field(default=None, description="GCS project ID")
    gcs_service_account_key: Optional[str] = Field(
        default=None,
        description="Path to GCS service account key JSON file"
    )
    gcs_public_base: Optional[str] = Field(
        default=None,
        description="Public base URL for uploaded files (if using CDN or public bucket)"
    )
    gcs_make_public: bool = Field(
        default=False,
        description="Automatically make uploaded blobs public"
    )
    gcs_signed_url_ttl_s: int = Field(default=3600, description="Signed URL TTL in seconds")

    # ===== Warmup Configuration =====
    warmup_on_start: bool = Field(default=False, description="Run warmup generation on startup")
    warmup_prompt: str = Field(default="A person walking in a park", description="Warmup prompt")
    warmup_image_url: Optional[str] = Field(default=None, description="Warmup image URL (optional)")

    # ===== Diagnostics / Test Endpoints =====
    test_r2_allow_any_path: bool = Field(
        default=False,
        description="Allow /test_r2 to upload any local mp4_path (unsafe for public deployments)"
    )
    test_r2_allowed_roots: str = Field(
        default="",
        description=f"Additional allowed roots for /test_r2 mp4_path (separated by '{os.pathsep}')"
    )

    # ===== Safety Limits (prevent abuse) =====
    max_width: int = Field(default=1280, description="Maximum allowed width")
    max_height: int = Field(default=1280, description="Maximum allowed height")
    max_fps: int = Field(default=30, description="Maximum allowed FPS")
    min_fps: int = Field(default=12, description="Minimum allowed FPS")
    max_steps: int = Field(default=30, description="Maximum allowed steps")
    min_steps: int = Field(default=1, description="Minimum allowed steps")
    max_duration_s: float = Field(default=10.0, description="Maximum duration in seconds")
    min_duration_s: float = Field(default=1.0, description="Minimum duration in seconds")

    def model_post_init(self, __context):
        """
        Post-initialization hook to auto-discover GCS credentials
        """
        if not self.gcs_service_account_key:
            # Check common locations for the key file
            potential_paths = [
                self.workdir / "gcs-service-account.json",
                Path("gcs-service-account.json"),
                Path("/app/gcs-service-account.json"),
            ]
            for path in potential_paths:
                if path.exists():
                    self.gcs_service_account_key = str(path.absolute())
                    break

    def setup_hf_env(self):
        """
        Set HuggingFace environment variables before importing transformers/diffusers
        This ensures models are cached in the correct locations
        """
        os.environ["HF_HOME"] = str(self.hf_home)
        os.environ["HF_HUB_CACHE"] = str(self.hf_hub_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(self.transformers_cache)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = self.hf_hub_enable_hf_transfer

        # Fix CUDA memory fragmentation for large models (23GB+ on 24GB GPUs)
        # This allows PyTorch to expand memory segments dynamically instead of pre-allocating
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

    def ensure_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.workdir,
            self.output_dir,
            self.hf_home,
            self.hf_hub_cache,
            self.transformers_cache,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
