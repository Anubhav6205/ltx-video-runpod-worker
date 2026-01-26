"""
FastAPI application entry point
Exposes HTTP endpoints for health checks and video generation
"""

# 1. Load configuration from volume (if present) BEFORE importing settings
# This allows credentials to be injected via /workspace/.env
try:
    from app.config_loader import load_volume_config
    load_volume_config()
except ImportError:
    pass

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import tempfile
import os

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.models import (
    GenerateClipRequest,
    GenerateClipResponse,
    HealthResponse,
    ReadyResponse,
    ErrorResponse,
    TestR2Response,
    TestR2Request,
)
from app.logging_config import setup_logging, get_logger, request_id_var, job_id_var
from app.bootstrap import init_model, is_ready, warmup_model
from app.infer import generate_clip
from app.storage import upload_video, upload_file_and_get_url_r2

# Initialize logging
setup_logging(settings.log_level)
logger = get_logger(__name__)

# Concurrency semaphore (prevents VRAM OOM on single GPU)
generation_semaphore: Optional[asyncio.Semaphore] = None


def _create_test_mp4(path: Path) -> None:
    try:
        import numpy as np
        import imageio.v2 as imageio

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        writer = imageio.get_writer(
            str(path),
            fps=1,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,
        )
        writer.append_data(frame)
        writer.close()
    except Exception:
        # Fallback: bytes-only file (still validates upload path + credentials)
        path.write_bytes(b"R2 upload test (not a real mp4)\n")


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _resolve_user_mp4_path(mp4_path: str) -> Path:
    path = Path(mp4_path)
    if not path.is_absolute():
        path = settings.workdir / path

    if not path.exists() or not path.is_file():
        raise ValueError(f"MP4 file not found: {path}")

    if path.suffix.lower() != ".mp4":
        raise ValueError(f"File must be a .mp4: {path}")

    allowed_roots = [settings.output_dir, settings.workdir]
    extra = (settings.test_r2_allowed_roots or "").strip()
    if extra:
        for part in extra.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            root = Path(part)
            if not root.is_absolute():
                root = settings.workdir / root
            allowed_roots.append(root)

    if not settings.test_r2_allow_any_path:
        if not any(_path_is_within(path, root) for root in allowed_roots):
            allowed_hint = f"{settings.output_dir} ; {settings.workdir}"
            if extra:
                allowed_hint = f"{allowed_hint} ; {extra}"
            raise ValueError(
                "mp4_path must be within OUTPUT_DIR/WORKDIR "
                f"(allowed roots: {allowed_hint}). "
                "To allow additional roots set TEST_R2_ALLOWED_ROOTS, "
                "or set TEST_R2_ALLOW_ANY_PATH=true (unsafe). "
                f"Got: {path}"
            )

    max_size_bytes = 1024 * 1024 * 1024  # 1 GiB safety cap for this endpoint
    size = path.stat().st_size
    if size > max_size_bytes:
        raise ValueError(f"MP4 file too large for test endpoint ({size} bytes)")

    return path


def _test_r2_upload_sync(request: Optional[TestR2Request]) -> tuple[str, str, str]:
    if not settings.r2_bucket:
        raise ValueError("R2_BUCKET is required to test R2 upload")

    job_id = request.job_id if request and request.job_id else str(uuid.uuid4())
    prefix = (settings.storage_key_prefix or "clips").strip().strip("/")
    key = f"{prefix}/{job_id}.mp4" if prefix else f"{job_id}.mp4"

    created_temp = False
    tmp_path: Optional[Path] = None
    if request and request.mp4_path:
        local_path = _resolve_user_mp4_path(request.mp4_path)
    else:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            prefix="r2-test-",
            suffix=".mp4",
            dir=str(settings.output_dir),
            delete=False,
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        _create_test_mp4(tmp_path)
        local_path = tmp_path
        created_temp = True

    try:
        url = upload_file_and_get_url_r2(local_path, key)
        return job_id, key, url
    finally:
        if created_temp and tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning(f"Failed to delete temp test file: {tmp_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Initializes model on startup and cleans up on shutdown
    """
    global generation_semaphore

    logger.info("=" * 60)
    logger.info("LTX-Video Worker Starting")
    logger.info("=" * 60)
    logger.info(f"Port: {settings.port}")
    logger.info(f"Max concurrent jobs: {settings.max_concurrent_jobs}")
    logger.info(f"Model: {settings.ltx_repo_id}")
    logger.info(f"HF Cache: {settings.hf_hub_cache}")

    # Initialize concurrency limiter
    generation_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
    logger.info(f"Concurrency semaphore initialized (limit: {settings.max_concurrent_jobs})")

    # Initialize model
    try:
        init_model()
        logger.info("Model loaded successfully!")

        # Optional warmup
        if settings.warmup_on_start:
            warmup_model()

    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}", exc_info=True)
        logger.warning("Server starting without model loaded - /ready will return false")

    logger.info("Server ready to accept requests")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="LTX-Video Worker",
    description="Image-to-video generation service using LTX-Video 2B distilled model",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request for tracing"""
    req_id = str(uuid.uuid4())
    request_id_var.set(req_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return clean JSON errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint
    Returns 200 if server is running
    """
    return HealthResponse(ok=True)


@app.get("/ready", response_model=ReadyResponse)
async def ready():
    """
    Readiness check endpoint
    Returns 200 only if model is loaded and ready for inference
    """
    model_ready = is_ready()
    return ReadyResponse(
        ready=model_ready,
        model_loaded=model_ready,
    )


@app.post("/test_r2", response_model=TestR2Response)
async def test_r2(request: Optional[TestR2Request] = None):
    """
    Upload a tiny test MP4 to Cloudflare R2 using the same env configuration
    as the generation flow, and return the resulting URL.
    """
    try:
        loop = asyncio.get_event_loop()
        job_id, key, url = await loop.run_in_executor(None, _test_r2_upload_sync, request)
        return TestR2Response(ok=True, job_id=job_id, bucket=settings.r2_bucket, key=key, url=url)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        logger.error(f"R2 test upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"R2 test upload failed: {str(e)}",
        )


@app.post("/generate_clip", response_model=GenerateClipResponse)
async def generate_clip_endpoint(request: GenerateClipRequest):
    """
    Generate a video clip from an image and text prompt

    Args:
        request: Generation parameters

    Returns:
        Job ID, public video URL, and metadata

    Raises:
        HTTPException: If generation fails or service is busy
    """
    # Check if model is ready
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded - please wait and try again",
        )

    # Check concurrency limit
    if generation_semaphore is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Concurrency semaphore not initialized",
        )

    # Try to acquire semaphore (non-blocking)
    if generation_semaphore.locked() and generation_semaphore._value == 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Server busy - max {settings.max_concurrent_jobs} concurrent job(s) allowed",
        )

    # Acquire semaphore and run generation
    async with generation_semaphore:
        try:
            logger.info("Starting generation request")
            logger.info(f"Image URL: {request.image_url}")
            logger.info(f"Prompt: {request.video_prompt}")

            # Run inference in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            job_id, local_path, meta = await loop.run_in_executor(
                None,  # Use default executor
                generate_clip,
                request,
            )

            # Set job_id in context for logging
            job_id_var.set(job_id)

            # Upload to S3/R2
            logger.info(f"Uploading video for job {job_id}")
            clip_url = await loop.run_in_executor(
                None,
                upload_video,
                job_id,
                local_path,
            )

            logger.info(f"Job {job_id} complete - URL: {clip_url}")

            return GenerateClipResponse(
                job_id=job_id,
                clip_url=clip_url,
                meta=meta,
            )

        except ValueError as e:
            # User error (invalid input, download failed, etc.)
            logger.warning(f"Generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        except Exception as e:
            # Internal error
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {str(e)}",
            )


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "LTX-Video Worker",
        "version": "0.1.0",
        "model": settings.ltx_repo_id,
        "ready": is_ready(),
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "generate": "/generate_clip",
            "test_r2": "/test_r2",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
