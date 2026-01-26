"""
Pydantic models for API request/response validation
Includes safety constraints and parameter validation
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator
from app.settings import settings


class GenerateClipRequest(BaseModel):
    """
    Request schema for video generation
    Validates all parameters against safety limits
    """

    image_url: HttpUrl = Field(
        ...,
        description="URL of the input image (must be publicly accessible)"
    )

    video_prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text prompt describing the desired video motion/content"
    )

    duration_s: float = Field(
        default=5.0,
        ge=settings.min_duration_s,
        le=settings.max_duration_s,
        description=f"Video duration in seconds ({settings.min_duration_s}-{settings.max_duration_s})"
    )

    model: Literal["ltxv-2b-distilled"] = Field(
        default="ltxv-2b-distilled",
        description="Model identifier (currently only ltxv-2b-distilled supported)"
    )

    width: int = Field(
        default=settings.default_width,
        ge=64,
        le=settings.max_width,
        description=f"Video width in pixels (64-{settings.max_width}, must be divisible by 32)"
    )

    height: int = Field(
        default=settings.default_height,
        ge=64,
        le=settings.max_height,
        description=f"Video height in pixels (64-{settings.max_height}, must be divisible by 32)"
    )

    fps: int = Field(
        default=settings.default_fps,
        ge=settings.min_fps,
        le=settings.max_fps,
        description=f"Frames per second ({settings.min_fps}-{settings.max_fps})"
    )

    steps: int = Field(
        default=settings.default_steps,
        ge=settings.min_steps,
        le=settings.max_steps,
        description=f"Number of inference steps ({settings.min_steps}-{settings.max_steps})"
    )

    guidance_scale: float = Field(
        default=settings.default_guidance_scale,
        ge=0.0,
        le=20.0,
        description="Guidance scale (1.0 recommended for distilled model)"
    )

    negative_prompt: Optional[str] = Field(
        default=settings.default_negative_prompt,
        max_length=500,
        description="Negative prompt to guide what to avoid"
    )

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducibility (optional)"
    )

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: int, info) -> int:
        """Ensure dimensions are divisible by 32 (required by VAE)"""
        if v % 32 != 0:
            raise ValueError(f"{info.field_name} must be divisible by 32, got {v}")
        return v

    @field_validator("video_prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Strip whitespace and ensure non-empty"""
        v = v.strip()
        if not v:
            raise ValueError("video_prompt cannot be empty")
        return v


class GenerationMeta(BaseModel):
    """Metadata about the generated video"""
    fps: int
    num_frames: int
    width: int
    height: int
    steps: int
    model: str
    seed: Optional[int] = None
    guidance_scale: float
    duration_s: float


class GenerateClipResponse(BaseModel):
    """Response schema for successful video generation"""
    job_id: str = Field(..., description="Unique job identifier")
    clip_url: str = Field(..., description="Public URL to download the generated MP4")
    meta: GenerationMeta = Field(..., description="Metadata about the generation")


class HealthResponse(BaseModel):
    """Health check response"""
    ok: bool = Field(default=True, description="Service is running")


class ReadyResponse(BaseModel):
    """Readiness check response"""
    ready: bool = Field(..., description="Model is loaded and ready for inference")
    model_loaded: bool = Field(..., description="Whether the model pipeline is loaded")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")
    job_id: Optional[str] = Field(default=None, description="Job ID if applicable")


class TestR2Response(BaseModel):
    """Response schema for R2 upload test"""
    ok: bool = Field(default=True, description="Whether the test succeeded")
    job_id: str = Field(..., description="Job ID used for the uploaded object name")
    bucket: str = Field(..., description="R2 bucket used for the test upload")
    key: str = Field(..., description="Object key written to R2")
    url: str = Field(..., description="Public/signed URL returned for the uploaded object")


class TestR2Request(BaseModel):
    """Request schema for R2 upload test"""
    mp4_path: Optional[str] = Field(
        default=None,
        description="Path to a local .mp4 file on the worker filesystem (defaults to a tiny generated test file)"
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Optional job_id to use for the uploaded filename (defaults to a random UUID)"
    )
