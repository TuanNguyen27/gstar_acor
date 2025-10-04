"""
Logging utilities for ACOR system
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "acor",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        format_string: Custom format string

    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger if not already done
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_string,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

    # Get specific logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)