"""
Configuration loader utility.
Loads configuration from a JSON credentials file (rc_creds.json) located in the volume.
This mimics the GCS service account key approach but for R2 credentials.
"""

import os
import json
from pathlib import Path
from app.logging_config import get_logger

logger = get_logger(__name__)

def load_volume_config():
    """
    Load credentials from rc_creds.json if it exists.
    Checks /workflow/rc_creds.json and /workspace/rc_creds.json.
    Injects found values into environment variables.
    """
    # Potential paths for the credentials file
    # User specifically asked for /workflow/rc_creds.json, but we also check /workspace
    # as that is the standard RunPod volume mount.
    potential_paths = [
        Path("/workflow/rc_creds.json"),
        Path("/workspace/rc_creds.json"),
    ]
    
    creds_path = None
    for path in potential_paths:
        if path.exists() and path.is_file():
            creds_path = path
            break
            
    if not creds_path:
        logger.info("No rc_creds.json found in volume, using standard environment variables")
        return

    logger.info(f"Loading credentials from: {creds_path}")
    
    try:
        with open(creds_path, "r") as f:
            creds = json.load(f)
            
        # Map JSON keys to environment variables
        # We support both direct env var names (R2_BUCKET) and lowercase (r2_bucket)
        loaded_count = 0
        for key, value in creds.items():
            env_key = key.upper()
            # Only inject if it looks like an R2 setting or related config
            if env_key.startswith("R2_") or env_key == "STORAGE_BACKEND":
                os.environ[env_key] = str(value)
                loaded_count += 1
                
        logger.info(f"Successfully loaded {loaded_count} configuration values from JSON")
        
        # Sanity check
        if os.getenv("R2_BUCKET"):
            logger.info(f"R2_BUCKET is set to: {os.getenv('R2_BUCKET')}")
        else:
            logger.warning("rc_creds.json loaded but R2_BUCKET is missing!")

    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from {creds_path}")
    except Exception as e:
        logger.error(f"Failed to load credentials: {str(e)}")