"""
Model bootstrapping and initialization
Downloads model weights and loads the LTX-Video pipeline at startup
Implements singleton pattern to ensure model is loaded only once
"""

import logging
import os
from pathlib import Path
from typing import Optional
import torch
from huggingface_hub import hf_hub_download

from app.settings import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Global pipeline instance (singleton)
PIPE: Optional[any] = None
_model_loaded: bool = False


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch dtype

    Args:
        dtype_str: String like "bfloat16", "float16", "float32"

    Returns:
        torch.dtype object
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        logger.warning(f"Unknown dtype '{dtype_str}', defaulting to bfloat16")
        dtype = torch.bfloat16
    return dtype


def get_cache_dir() -> str:
    """
    Find the correct HuggingFace cache directory with fallback support.
    Checks multiple locations to handle Docker ENV mismatches gracefully.

    Priority:
    1. /workspace/models/hf_hub (preferred - volume mount)
    2. ENV variable HF_HUB_CACHE
    3. /models/hf_hub (fallback for old Docker images)

    Returns:
        Path to cache directory with existing weights, or default location
    """
    candidates = [
        Path("/workspace/models/hf_hub"),  # Preferred (volume)
        Path(os.getenv("HF_HUB_CACHE", "/workspace/models/hf_hub")),  # ENV override
        Path("/models/hf_hub"),  # Fallback (old Docker ENV)
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for path in candidates:
        path_str = str(path)
        if path_str not in seen:
            seen.add(path_str)
            unique_candidates.append(path)

    logger.info("Searching for cached model weights...")
    for cache_dir in unique_candidates:
        if cache_dir.exists():
            # Check if LTX-Video model weights exist
            model_files = list(cache_dir.glob("models--Lightricks--LTX-Video/**/*.safetensors"))
            if model_files:
                logger.info(f"✅ Found cached weights at: {cache_dir}")
                logger.info(f"   Model files: {len(model_files)} safetensors found")
                return str(cache_dir)
            else:
                logger.info(f"   {cache_dir} exists but no model weights found")
        else:
            logger.info(f"   {cache_dir} does not exist")

    # No cached weights found - use preferred location
    default = "/workspace/models/hf_hub"
    logger.warning(f"⚠️  No cached weights found in any location!")
    logger.warning(f"   Will attempt to use: {default}")
    logger.warning(f"   Note: Downloads are DISABLED - ensure volume is mounted correctly")
    return default


def init_model() -> None:
    """
    Initialize the LTX-Video pipeline
    Downloads model weights and loads pipeline to GPU
    This function is idempotent - safe to call multiple times

    Raises:
        RuntimeError: If model initialization fails
    """
    global PIPE, _model_loaded

    if _model_loaded and PIPE is not None:
        logger.info("Model already loaded, skipping initialization")
        return

    try:
        logger.info("Initializing LTX-Video model...")
        logger.info(f"Repository: {settings.ltx_repo_id}")
        logger.info(f"Transformer file: {settings.ltx_transformer_filename}")
        logger.info(f"Torch dtype: {settings.torch_dtype}")

        # Ensure HF environment is set up
        settings.setup_hf_env()
        settings.ensure_directories()

        # Use smart cache directory detection with fallback
        cache_dir = get_cache_dir()
        logger.info(f"Using cache directory: {cache_dir}")

        # Import here after env vars are set
        # Using LTXImageToVideoPipeline for image-to-video generation
        # See: https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video
        from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel

        # Check for transformer weights (local only - no downloads)
        logger.info("Loading transformer weights from local cache...")
        transformer_path = None
        try:
            transformer_path = hf_hub_download(
                repo_id=settings.ltx_repo_id,
                filename=settings.ltx_transformer_filename,
                cache_dir=cache_dir,
                local_files_only=True,  # CRITICAL: Never download, fail fast
            )
            logger.info(f"✅ Transformer weights found: {transformer_path}")
        except Exception as e:
            logger.error(f"❌ Failed to find transformer weights in cache!")
            logger.error(f"   Cache directory: {cache_dir}")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   SOLUTION: Ensure volume is mounted to /workspace with pre-cached weights")
            raise RuntimeError(
                f"Model weights not found in cache. Downloads are disabled. "
                f"Please ensure volume with model weights is mounted to /workspace. "
                f"Searched: {cache_dir}"
            )

        # Load pipeline from pretrained (local only - no downloads)
        # Recommended dtype: torch.bfloat16 (supports float16, float32 for VAE/text encoder)
        logger.info("Loading LTX-Video pipeline from local cache...")
        torch_dtype = get_torch_dtype(settings.torch_dtype)

        try:
            # Manually load transformer if we found a specific file
            # This handles FP8 loading or custom checkpoint files
            transformer = None
            if transformer_path:
                logger.info(f"Loading custom transformer from: {transformer_path}")
                
                # Determine dtype for transformer
                # ALWAYS load in bfloat16 first to ensure structure compatibility
                # We will quantize to FP8 in-place using torchao if needed
                transformer = LTXVideoTransformer3DModel.from_single_file(
                    transformer_path, 
                    torch_dtype=torch_dtype
                )

                if settings.use_fp8:
                    logger.info("Applying FP8 quantization via torchao...")
                    try:
                        from torchao.quantization import quantize_, float8_weight_only
                        quantize_(transformer, float8_weight_only())
                        logger.info("✅ Transformer quantized to FP8 (torchao)")
                    except ImportError:
                        logger.error("❌ torchao not installed! cannot use FP8. Falling back to BF16 (may OOM).")
                    except Exception as e:
                        logger.error(f"❌ FP8 quantization failed: {e}")

            PIPE = LTXImageToVideoPipeline.from_pretrained(
                settings.ltx_repo_id,
                transformer=transformer,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                local_files_only=True,  # CRITICAL: Never download
            )
            logger.info("✅ Pipeline loaded successfully from local cache")
        except Exception as e:
            logger.error(f"❌ Failed to load pipeline from cache!")
            logger.error(f"   Cache directory: {cache_dir}")
            logger.error(f"   Error: {str(e)}")
            raise RuntimeError(
                f"Pipeline loading failed. Ensure all model components are cached. "
                f"Downloads are disabled for cost protection."
            )

        # Move to CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available - GPU is required!")

        logger.info(f"Moving pipeline to CUDA (GPU: {torch.cuda.get_device_name(0)})")
        PIPE = PIPE.to("cuda")

        # Enable memory optimizations
        logger.info("Enabling memory-efficient attention...")
        PIPE.enable_model_cpu_offload()  # Offload modules not in use

        _model_loaded = True
        logger.info("Model initialization complete!")

        # Log VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}", exc_info=True)
        _model_loaded = False
        PIPE = None
        raise RuntimeError(f"Model initialization failed: {str(e)}")


def is_ready() -> bool:
    """
    Check if the model is loaded and ready for inference

    Returns:
        True if model is ready, False otherwise
    """
    return _model_loaded and PIPE is not None


def get_pipeline():
    """
    Get the loaded pipeline instance

    Returns:
        LTXImageToVideoPipeline instance

    Raises:
        RuntimeError: If pipeline is not loaded
    """
    if not is_ready():
        raise RuntimeError("Pipeline not initialized - call init_model() first")
    return PIPE


def warmup_model():
    """
    Run a warmup generation to pre-compile kernels and cache
    This can reduce latency on the first real request
    Only runs if WARMUP_ON_START is enabled in settings
    """
    if not settings.warmup_on_start:
        logger.info("Warmup disabled (WARMUP_ON_START=false)")
        return

    if not is_ready():
        logger.warning("Cannot warmup - model not loaded")
        return

    try:
        logger.info("Running warmup generation...")

        # Create a small test image if no warmup image provided
        from PIL import Image
        import numpy as np

        if settings.warmup_image_url:
            from app.utils import download_image
            test_image = download_image(settings.warmup_image_url, timeout_s=30)
        else:
            # Generate random test image
            test_image = Image.fromarray(
                np.random.randint(0, 255, (512, 704, 3), dtype=np.uint8)
            )

        # Run minimal generation
        pipeline = get_pipeline()
        with torch.inference_mode():
            _ = pipeline(
                prompt=settings.warmup_prompt,
                image=test_image,
                num_frames=25,  # 1 second at 24fps
                num_inference_steps=4,  # Minimal steps
                guidance_scale=1.0,
                height=512,
                width=704,
            )

        logger.info("Warmup complete!")

        # Log VRAM after warmup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory after warmup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {str(e)}")
