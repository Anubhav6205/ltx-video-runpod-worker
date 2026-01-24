"""
FastAPI application entry point
Exposes HTTP endpoints for health checks and video generation
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

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
)
from app.logging_config import setup_logging, get_logger, request_id_var, job_id_var
from app.bootstrap import init_model, is_ready, warmup_model
from app.infer import generate_clip
from app.storage import upload_video

# Initialize logging
setup_logging(settings.log_level)
logger = get_logger(__name__)

# Concurrency semaphore (prevents VRAM OOM on single GPU)
generation_semaphore: Optional[asyncio.Semaphore] = None


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
