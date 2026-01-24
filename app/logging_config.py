"""
Structured logging configuration with JSON-like output
Includes request ID tracking and sensitive data filtering
"""

import logging
import sys
import time
from typing import Any, Dict, Optional
from contextvars import ContextVar
import json

# Context variable for tracking request/job IDs across async calls
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
job_id_var: ContextVar[Optional[str]] = ContextVar("job_id", default=None)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured log records
    Filters sensitive data and includes context IDs
    """

    SENSITIVE_KEYS = {"password", "token", "secret", "api_key", "access_key", "secret_key"}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON-like string"""

        # Base log structure
        log_dict: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request/job IDs from context if available
        request_id = request_id_var.get()
        if request_id:
            log_dict["request_id"] = request_id

        job_id = job_id_var.get()
        if job_id:
            log_dict["job_id"] = job_id

        # Add extra fields from record, filtering sensitive data
        if hasattr(record, "extra_data"):
            extra = self._filter_sensitive(record.extra_data)
            log_dict.update(extra)

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict)

    def _filter_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out sensitive keys from log data"""
        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive(value)
            else:
                filtered[key] = value
        return filtered


def setup_logging(log_level: str = "INFO"):
    """
    Configure application logging with structured output

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with structured formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: int, message: str, **kwargs):
    """
    Log a message with additional context data

    Args:
        logger: Logger instance
        level: Log level (logging.INFO, etc.)
        message: Log message
        **kwargs: Additional context data to include
    """
    extra_record = type("ExtraRecord", (), {"extra_data": kwargs})()
    logger.log(level, message, extra={"extra_data": kwargs})
