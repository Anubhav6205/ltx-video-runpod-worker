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
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Server Configuration =====
    port: int = Field(default=8000, description="HTTP server port")
    log_level: str = Field(default="INFO", description="Logging level")
    workdir: Path = Field(default=Path("/workspace"), description="Working directory")
    output_dir: Path = Field(default=Path("/workspace/outputs"), description="Video output directory")

    # ===== Concurrency & Timeouts =====
    max_concurrent_jobs: int = Field(default=1, description="Max simultaneous generations (1 = safest on single GPU)")
    image_download_timeout_s: int = Field(default=30, description="Timeout for downloading input images")
    generate_timeout_s: int = Field(default=900, description="Timeout for video generation (4090 + offload can be slow)")

    # ===== Hugging Face / Caching =====
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token (optional for public models)")
    hf_home: Path = Field(default=Path("/workspace/models/hf_home"), description="HuggingFace home directory")
    hf_hub_cache: Path = Field(default=Path("/workspace/models/hf_hub"), description="HuggingFace Hub cache directory")
    transformers_cache: Path = Field(default=Path("/workspace/models/hf_transformers"), description="Transformers cache")

    transformers_offline: bool = Field(
        default=True,
        description="Sets TRANSFORMERS_OFFLINE=1 to avoid network calls (recommended when weights are pre-cached).",
    )
    hf_hub_offline: bool = Field(
        default=False,
        description="Sets HF_HUB_OFFLINE=1 (enable only if you want strict hub offline).",
    )
    enable_hf_transfer: bool = Field(
        default=False,
        description="Sets HF_HUB_ENABLE_HF_TRANSFER=1 (only useful if downloading from Hub).",
    )

    # ===== LTX-Video Model Configuration =====
    ltx_repo_id: str = Field(default="Lightricks/LTX-Video", description="HuggingFace repository ID")
    ltx_transformer_filename: str = Field(
        default="ltxv-13b-0.9.8-distilled-fp8.safetensors",
        description="Transformer weights filename",
    )
    torch_dtype: str = Field(default="bfloat16", description="Torch dtype (bfloat16, float16, float32)")

    # Pre-quantized FP8 weights: do NOT re-quantize in production.
    enable_torchao_fp8_quantization: bool = Field(
        default=False,
        description="Only for non-FP8 checkpoints; keep False for pre-quantized fp8 safetensors.",
    )

    # RTX 4090 (24GB): stability-first default is to offload.
    enable_cpu_offload: bool = Field(
        default=True,
        description="Enable Diffusers CPU offload to reduce VRAM OOM risk on 24GB GPUs.",
    )

    # ===== Default Generation Parameters =====
    default_width: int = Field(default=704, description="Default video width")
    default_height: int = Field(default=512, description="Default video height")
    default_fps: int = Field(default=24, description="Default fps")
    default_steps: int = Field(default=8, description="Default steps (distilled model often best 4-8)")
    default_guidance_scale: float = Field(default=1.0, description="Default guidance (distilled)")
    default_negative_prompt: str = Field(
        default="worst quality, blurry, jittery, distorted, watermark, text",
        description="Default negative prompt",
    )

    # ===== Storage Configuration =====
    storage_backend: str = Field(default="r2", description="Storage backend: r2 or gcs")
    storage_key_prefix: str = Field(default="clips", description="Prefix/folder for uploaded videos")

    # --- Cloudflare R2 (S3-compatible) ---
    r2_bucket: str = Field(default="", description="Cloudflare R2 bucket name")
    r2_account_id: Optional[str] = Field(default=None, description="Cloudflare account ID")
    r2_endpoint_url: Optional[str] = Field(default=None, description="R2 S3 endpoint URL")
    r2_access_key_id: Optional[str] = Field(default=None, description="R2 access key ID")
    r2_secret_access_key: Optional[str] = Field(default=None, description="R2 secret access key")
    r2_region: str = Field(default="auto", description="S3 region (use 'auto' for R2)")
    r2_public_base: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_PUBLIC_BASE", "R2_PUBLIC_BASE_URL"),
        description="Public base URL (supports {bucket} placeholder)",
    )
    r2_signed_url_ttl_s: int = Field(
        default=3600,
        validation_alias=AliasChoices("R2_SIGNED_URL_TTL_S", "R2_PRESIGN_TTL_S"),
        description="Signed URL TTL seconds",
    )

    # --- Google Cloud Storage (optional) ---
    gcs_bucket: str = Field(default="", description="GCS bucket name")
    gcs_project_id: Optional[str] = Field(default=None, description="GCS project ID")
    gcs_service_account_key: Optional[str] = Field(default=None, description="Path to GCS key JSON")
    gcs_public_base: Optional[str] = Field(default=None, description="Public base URL")
    gcs_make_public: bool = Field(default=False, description="Make uploaded blobs public")
    gcs_signed_url_ttl_s: int = Field(default=3600, description="Signed URL TTL seconds")

    # ===== Warmup Configuration =====
    warmup_on_start: bool = Field(default=False, description="Run warmup generation on startup")
    warmup_prompt: str = Field(default="A person walking in a park", description="Warmup prompt")
    warmup_image_url: Optional[str] = Field(default=None, description="Warmup image URL")

    # ===== Diagnostics / Test Endpoints =====
    test_r2_allow_any_path: bool = Field(default=False, description="Unsafe: allow any local mp4_path")
    test_r2_allowed_roots: str = Field(
        default="",
        description=f"Allowed roots for /test_r2 mp4_path (separated by '{os.pathsep}')",
    )

    # ===== Safety Limits =====
    max_width: int = Field(default=1280, description="Max width")
    max_height: int = Field(default=1280, description="Max height")
    max_fps: int = Field(default=30, description="Max fps")
    min_fps: int = Field(default=12, description="Min fps")
    max_steps: int = Field(default=30, description="Max steps")
    min_steps: int = Field(default=1, description="Min steps")
    max_duration_s: float = Field(default=10.0, description="Max duration seconds")
    min_duration_s: float = Field(default=1.0, description="Min duration seconds")

    # ===== CUDA allocator tuning =====
    pytorch_cuda_alloc_conf: str = Field(
        default="expandable_segments:True",
        description="PYTORCH_CUDA_ALLOC_CONF to reduce fragmentation.",
    )

    def model_post_init(self, __context):
        if not self.gcs_service_account_key:
            for path in [
                self.workdir / "gcs-service-account.json",
                Path("gcs-service-account.json"),
                Path("/app/gcs-service-account.json"),
            ]:
                if path.exists():
                    self.gcs_service_account_key = str(path.absolute())
                    break

    def setup_hf_env(self):
        os.environ["HF_HOME"] = str(self.hf_home)
        os.environ["HF_HUB_CACHE"] = str(self.hf_hub_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(self.transformers_cache)

        if self.transformers_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

        if self.hf_hub_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if self.enable_hf_transfer else "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.pytorch_cuda_alloc_conf

        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

    def ensure_directories(self):
        for directory in [
            self.workdir,
            self.output_dir,
            self.hf_home,
            self.hf_hub_cache,
            self.transformers_cache,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
