"""Logging utilities for PDE simulations."""

import logging
import sys
from pathlib import Path


class TeeWriter:
    """Writer that outputs to multiple destinations (tee-like behavior)."""

    def __init__(self, *writers):
        self.writers = writers

    def write(self, message):
        for writer in self.writers:
            if writer is not None:
                writer.write(message)
                # Flush immediately for real-time output
                try:
                    writer.flush()
                except (AttributeError, OSError):
                    pass

    def flush(self):
        for writer in self.writers:
            if writer is not None:
                try:
                    writer.flush()
                except (AttributeError, OSError):
                    pass


# Store original stdout/stderr for restoration
_original_stdout = None
_original_stderr = None
_log_file_handle = None


def setup_logging(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    console: bool = True,
    capture_stdout: bool = False,
) -> logging.Logger:
    """Configure logging for PDE simulations.

    Args:
        log_file: Optional path to log file. If provided, logs will be written to this file.
        level: Logging level (default: INFO).
        console: Whether to also log to console (default: True).
        capture_stdout: If True and log_file is provided, also capture stdout/stderr to log file.
            This is useful for capturing py-pde progress output.

    Returns:
        Configured logger instance.
    """
    global _original_stdout, _original_stderr, _log_file_handle

    logger = logging.getLogger("pde_sim")
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up stdout/stderr capture BEFORE creating console handler
    # so the handler uses the TeeWriter
    if log_file is not None and capture_stdout:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Store originals for restoration
        _original_stdout = sys.stdout
        _original_stderr = sys.stderr

        # Open log file for stdout/stderr capture
        _log_file_handle = open(log_path, "w")

        # Create tee writers
        if console:
            sys.stdout = TeeWriter(_original_stdout, _log_file_handle)
            sys.stderr = TeeWriter(_original_stderr, _log_file_handle)
        else:
            # Only write to file, not console
            sys.stdout = TeeWriter(_log_file_handle)
            sys.stderr = TeeWriter(_log_file_handle)

    # Add console handler if requested (now uses TeeWriter if capture_stdout)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is provided (but not if capturing stdout,
    # since TeeWriter already writes to file)
    if log_file is not None and not capture_stdout:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def restore_stdout():
    """Restore original stdout/stderr after capturing."""
    global _original_stdout, _original_stderr, _log_file_handle

    if _original_stdout is not None:
        sys.stdout = _original_stdout
        _original_stdout = None

    if _original_stderr is not None:
        sys.stderr = _original_stderr
        _original_stderr = None

    if _log_file_handle is not None:
        try:
            _log_file_handle.close()
        except (OSError, AttributeError):
            pass
        _log_file_handle = None


def get_logger() -> logging.Logger:
    """Get the pde_sim logger instance.

    Returns:
        The pde_sim logger. If setup_logging hasn't been called,
        returns an unconfigured logger.
    """
    return logging.getLogger("pde_sim")
