"""
Model bootstrapping and initialization
Downloads model weights and loads the LTX-Video pipeline at startup
Implements singleton pattern to ensure model is loaded only once
"""

import gc
import logging
import os
from contextlib import nullcontext
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


def _try_enable(pipe, method_name: str, *args, **kwargs) -> bool:
    """
    Best-effort enable of optional diffusers pipeline optimizations.
    Returns True if successfully enabled, False otherwise.
    """
    if not hasattr(pipe, method_name):
        logger.debug(f"Pipeline does not support {method_name}")
        return False
    try:
        getattr(pipe, method_name)(*args, **kwargs)
        logger.info(f"Enabled {method_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to enable {method_name}: {e}")
        return False


def _apply_memory_optimizations(pipe, total_vram_gb: float) -> None:
    """
    Enable memory-saving features that should not change output quality.
    These trade latency for lower peak VRAM.
    """
    # Prefer xFormers attention when available; fallback to attention slicing.
    enabled_xformers = False
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            enabled_xformers = True
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Failed to enable xformers attention: {e}")

    if not enabled_xformers:
        _try_enable(pipe, "enable_attention_slicing", "auto")

    # VAE slicing reduces decoder peak memory.
    _try_enable(pipe, "enable_vae_slicing")

    # VAE tiling helps for larger resolutions or tight VRAM budgets.
    if total_vram_gb <= 32:
        _try_enable(pipe, "enable_vae_tiling")


def _cast_non_transformer_modules(pipe, dtype: torch.dtype) -> None:
    """
    For FP8 transformers, avoid any whole-pipeline dtype conversion that might upcast the transformer.
    Instead, cast known non-transformer modules to the requested dtype.
    """
    for name in ("vae", "text_encoder", "text_encoder_2", "image_encoder"):
        mod = getattr(pipe, name, None)
        if mod is None:
            continue
        if not isinstance(mod, torch.nn.Module):
            continue
        try:
            mod.to(dtype=dtype)
            logger.info(f"Casted {name} to {dtype}")
        except Exception as e:
            logger.warning(f"Failed to cast {name} to {dtype}: {e}")


def ensure_non_transformer_modules_dtype(pipe, dtype: torch.dtype) -> None:
    """
    Best-effort: ensure non-transformer modules (VAE/text encoders) match requested dtype.
    Useful as a runtime recovery step if a request hits a BF16/FP32 matmul mismatch.
    """
    _cast_non_transformer_modules(pipe, dtype)


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
            transformer_dtype = None
            if transformer_path:
                logger.info(f"Loading custom transformer from: {transformer_path}")

                # Check if this is a pre-quantized FP8 model
                is_pre_quantized_fp8 = "fp8" in settings.ltx_transformer_filename.lower()

                if is_pre_quantized_fp8:
                    # Model is ALREADY in FP8 format - load without dtype conversion
                    # CRITICAL: Don't specify torch_dtype to preserve native FP8 format
                    logger.info("Detected pre-quantized FP8 model - loading in native FP8 format")
                    logger.info("⚠️  Loading without dtype conversion to prevent BF16/FP8 mismatch")

                    # Aggressively clear GPU memory before loading to prevent OOM
                    if torch.cuda.is_available():
                        # Multiple passes to ensure thorough cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.empty_cache()

                        # Log memory status
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - allocated
                        logger.info(f"Pre-load GPU memory: {allocated:.2f}GB allocated, {free:.2f}GB free")

                    # Load without torch_dtype to preserve FP8 weights
                    # Use low_cpu_mem_usage to minimize memory overhead (1x model size instead of 2x)
                    # Load to CPU first - CPU offload will manage GPU placement
                    transformer = LTXVideoTransformer3DModel.from_single_file(
                        transformer_path,
                        low_cpu_mem_usage=True
                    )
                    logger.info("✅ Transformer loaded to CPU (CPU offload will manage GPU placement)")

                    transformer_dtype = None  # Signal that transformer is FP8, don't convert
                    logger.info("✅ Pre-quantized FP8 transformer loaded in native format")
                else:
                    # Model is NOT pre-quantized - load in bfloat16 and optionally quantize
                    logger.info("Loading non-quantized model in bfloat16")

                    # Aggressively clear GPU memory before loading to prevent OOM
                    if torch.cuda.is_available():
                        # Multiple passes to ensure thorough cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.empty_cache()

                        # Log memory status
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - allocated
                        logger.info(f"Pre-load GPU memory: {allocated:.2f}GB allocated, {free:.2f}GB free")

                    transformer = LTXVideoTransformer3DModel.from_single_file(
                        transformer_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True  # Minimize memory overhead during loading
                    )
                    transformer_dtype = torch_dtype

                    # Only apply torchao quantization if requested AND model is not already FP8
                    use_fp8_env = os.getenv("USE_FP8", "").strip().lower() in {"1", "true", "yes", "on"}
                    if settings.enable_torchao_fp8_quantization or use_fp8_env:
                        logger.info("Applying FP8 quantization via torchao...")
                        try:
                            from torchao.quantization import quantize_, float8_weight_only
                            quantize_(transformer, float8_weight_only())
                            transformer_dtype = None  # Now it's FP8
                            logger.info("✅ Transformer quantized to FP8 (torchao)")
                        except ImportError:
                            logger.error("❌ torchao not installed! cannot use FP8. Falling back to BF16 (may OOM).")
                        except Exception as e:
                            logger.error(f"❌ FP8 quantization failed: {e}")

            # Load pipeline with proper dtype handling for FP8 compatibility
            # For FP8 transformers: VAE and text encoder use bfloat16, transformer stays FP8
            logger.info("Loading pipeline components...")
            if transformer_dtype is None:
                logger.info("Using FP8 transformer with bfloat16 VAE/text encoder")

            # IMPORTANT: When using an FP8 transformer (pre-quantized or torchao-quantized),
            # avoid passing torch_dtype to from_pretrained because some code paths may apply
            # dtype conversion broadly and inadvertently upcast the FP8 transformer.
            if transformer is not None and transformer_dtype is None:
                PIPE = LTXImageToVideoPipeline.from_pretrained(
                    settings.ltx_repo_id,
                    transformer=transformer,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                _cast_non_transformer_modules(PIPE, torch_dtype)
            else:
                PIPE = LTXImageToVideoPipeline.from_pretrained(
                    settings.ltx_repo_id,
                    transformer=transformer,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
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

        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available - GPU is required!")

        # Smart GPU memory management based on available VRAM
        # 13B FP8 model needs ~26GB total (23GB transformer + 2GB VAE + 1GB text encoder)
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        logger.info(f"GPU: {gpu_name}, Total VRAM: {total_vram_gb:.1f}GB")

        # Apply safe memory optimizations before placing on GPU
        _apply_memory_optimizations(PIPE, total_vram_gb)

        if settings.enable_cpu_offload:
            # Prefer offload when enabled to preserve VRAM headroom for activations
            logger.info("CPU offload enabled by settings")
            try:
                if total_vram_gb <= 32 and hasattr(PIPE, "enable_sequential_cpu_offload"):
                    PIPE.enable_sequential_cpu_offload()
                    logger.info("CPU sequential offload enabled")
                else:
                    PIPE.enable_model_cpu_offload()
                    logger.info("CPU offload enabled - only active model kept on GPU")
            except Exception as e:
                logger.warning(f"CPU offload failed: {e}")
        else:
            # Only attempt full GPU load when there's ample headroom
            full_gpu_threshold_gb = 40.0
            if total_vram_gb >= full_gpu_threshold_gb:
                logger.info(f"Large GPU detected ({total_vram_gb:.1f}GB) - attempting full GPU load")
                try:
                    PIPE = PIPE.to("cuda")
                    logger.info("Pipeline fully loaded to GPU - no CPU offload needed")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("Full GPU load failed, falling back to CPU offload")
                        torch.cuda.empty_cache()
                        PIPE.enable_model_cpu_offload()
                        logger.info("CPU offload enabled - only active model kept on GPU")
                    else:
                        raise
            else:
                logger.info(f"GPU ({total_vram_gb:.1f}GB) below full-load threshold - enabling CPU offload")
                PIPE.enable_model_cpu_offload()
                logger.info("CPU offload enabled - only active model kept on GPU")

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
            torch_dtype = get_torch_dtype(settings.torch_dtype)
            use_autocast = torch_dtype in (torch.float16, torch.bfloat16)
            autocast_ctx = torch.autocast("cuda", dtype=torch_dtype) if use_autocast else nullcontext()
            with autocast_ctx:
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
