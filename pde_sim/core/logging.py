"""Logging utilities for PDE simulations."""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure logging for PDE simulations.

    Args:
        log_file: Optional path to log file. If provided, logs will be written to this file.
        level: Logging level (default: INFO).
        console: Whether to also log to console (default: True).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("pde_sim")
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is provided
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the pde_sim logger instance.

    Returns:
        The pde_sim logger. If setup_logging hasn't been called,
        returns an unconfigured logger.
    """
    return logging.getLogger("pde_sim")
