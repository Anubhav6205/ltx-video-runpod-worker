"""
Utility functions for image download, file handling, and safety checks
"""

import re
import logging
from pathlib import Path
from typing import Optional
from io import BytesIO

import requests
from PIL import Image

from app.logging_config import get_logger

logger = get_logger(__name__)


def download_image(url: str, timeout_s: int = 30) -> Image.Image:
    """
    Download an image from a URL and return as PIL Image

    Args:
        url: Image URL (must be publicly accessible)
        timeout_s: Request timeout in seconds

    Returns:
        PIL Image in RGB mode

    Raises:
        ValueError: If download fails or image is invalid
    """
    try:
        # Log sanitized URL (domain only for privacy)
        sanitized_url = sanitize_url_for_logging(url)
        logger.info(f"Downloading image from {sanitized_url}")

        # Download with timeout
        response = requests.get(
            url,
            timeout=timeout_s,
            headers={"User-Agent": "LTX-Worker/0.1"}
        )
        response.raise_for_status()

        # Validate content type
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"URL did not return an image (content-type: {content_type})")

        # Load and convert to RGB
        image = Image.open(BytesIO(response.content))
        if image.mode != "RGB":
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")

        logger.info(f"Downloaded image: {image.size[0]}x{image.size[1]}")
        return image

    except requests.RequestException as e:
        raise ValueError(f"Failed to download image: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def sanitize_url_for_logging(url: str) -> str:
    """
    Sanitize URL for logging by removing query parameters and keeping only domain/path
    This prevents logging signed URLs or sensitive tokens

    Args:
        url: Full URL

    Returns:
        Sanitized URL (scheme + domain + path only)
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Return scheme + netloc + path (no query params)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except:
        return "[URL]"


def safe_filename(name: str) -> str:
    """
    Convert a string to a safe filename
    Removes/replaces problematic characters

    Args:
        name: Input string

    Returns:
        Safe filename string
    """
    # Remove/replace unsafe characters
    safe = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    # Limit length
    safe = safe[:200]
    return safe


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max

    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes as human-readable string

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"
