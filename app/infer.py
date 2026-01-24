"""
Video generation inference logic
Handles the full pipeline from image download to video export
"""

import uuid
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import torch
from PIL import Image

from app.settings import settings
from app.models import GenerateClipRequest, GenerationMeta
from app.bootstrap import get_pipeline
from app.utils import download_image, ensure_directory
from app.logging_config import get_logger

logger = get_logger(__name__)


def generate_clip(request: GenerateClipRequest) -> Tuple[str, Path, GenerationMeta]:
    """
    Generate a video clip from an image and prompt

    Args:
        request: Validated generation request

    Returns:
        Tuple of (job_id, local_mp4_path, metadata)

    Raises:
        ValueError: If generation fails
        RuntimeError: If pipeline not ready
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Starting generation job {job_id}")
    logger.info(f"Prompt: {request.video_prompt}")
    logger.info(f"Params: {request.width}x{request.height} @ {request.fps}fps, {request.duration_s}s, {request.steps} steps")

    try:
        # 1. Download input image
        logger.info(f"Downloading input image...")
        input_image = download_image(
            str(request.image_url),
            timeout_s=settings.image_download_timeout_s
        )

        # Resize if needed (VAE requires specific dimensions)
        if input_image.size != (request.width, request.height):
            logger.info(f"Resizing image from {input_image.size} to {request.width}x{request.height}")
            input_image = input_image.resize((request.width, request.height), Image.Resampling.LANCZOS)

        # 2. Calculate number of frames
        num_frames = int(request.duration_s * request.fps) + 1
        logger.info(f"Generating {num_frames} frames ({request.duration_s}s @ {request.fps}fps)")

        # 3. Set up generation parameters
        generator = None
        if request.seed is not None:
            logger.info(f"Using seed: {request.seed}")
            generator = torch.Generator(device="cuda").manual_seed(request.seed)

        # 4. Run inference
        logger.info("Running LTX-Video inference...")
        pipeline = get_pipeline()

        with torch.inference_mode():
            # Use autocast for mixed precision if using float16
            context = torch.cuda.amp.autocast(enabled=(settings.torch_dtype == "float16"))

            with context:
                # Using recommended parameters from LTX-Video 0.9.1+ documentation
                # See: https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video
                result = pipeline(
                    prompt=request.video_prompt,
                    negative_prompt=request.negative_prompt,
                    image=input_image,
                    num_frames=num_frames,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    height=request.height,
                    width=request.width,
                    frame_rate=request.fps,  # Added for temporal coordinates normalization
                    decode_timestep=0.05,     # Recommended for timestep-aware VAE variants (0.9.1+)
                    decode_noise_scale=0.025, # Recommended noise interpolation at decode time
                    generator=generator,
                )

        logger.info("Inference complete")

        # 5. Export video frames to MP4
        output_path = ensure_directory(settings.output_dir) / f"{job_id}.mp4"
        logger.info(f"Exporting video to {output_path}")

        # Get video frames from result
        video_frames = result.frames[0]  # First (and only) video in batch

        # Export using diffusers utility
        from diffusers.utils import export_to_video
        export_to_video(
            video_frames,
            str(output_path),
            fps=request.fps
        )

        file_size = output_path.stat().st_size
        logger.info(f"Video exported successfully: {file_size / 1024 / 1024:.2f} MB")

        # 6. Create metadata
        meta = GenerationMeta(
            fps=request.fps,
            num_frames=num_frames,
            width=request.width,
            height=request.height,
            steps=request.steps,
            model=request.model,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            duration_s=request.duration_s,
        )

        # Log VRAM usage after generation
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        logger.info(f"Generation job {job_id} complete!")
        return job_id, output_path, meta

    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {str(e)}", exc_info=True)
        raise ValueError(f"Video generation failed: {str(e)}")
