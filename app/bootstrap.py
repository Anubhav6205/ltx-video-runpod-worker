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

        # Import here after env vars are set
        from diffusers import LTXImageToVideoPipeline

        # Download transformer weights explicitly to ensure caching
        logger.info("Downloading transformer weights...")
        transformer_path = hf_hub_download(
            repo_id=settings.ltx_repo_id,
            filename=settings.ltx_transformer_filename,
            token=settings.hf_token,
            cache_dir=str(settings.hf_hub_cache),
        )
        logger.info(f"Transformer weights cached at: {transformer_path}")

        # Load pipeline from pretrained
        logger.info("Loading LTX-Video pipeline...")
        torch_dtype = get_torch_dtype(settings.torch_dtype)

        PIPE = LTXImageToVideoPipeline.from_pretrained(
            settings.ltx_repo_id,
            torch_dtype=torch_dtype,
            cache_dir=str(settings.hf_hub_cache),
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
