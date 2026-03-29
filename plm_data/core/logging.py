"""Centralized logging setup for PLM-data simulations."""

import logging
import sys
from pathlib import Path

from mpi4py import MPI

_PACKAGE_LOGGER_NAME = "plm_data"
_LOG_FILE_NAME = "simulation.log"


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the plm_data namespace."""
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def setup_logging(output_dir: Path, console_level: int = logging.INFO) -> None:
    """Configure file and console logging for a simulation run.

    Args:
        output_dir: Directory for the log file (simulation.log).
        console_level: Logging level for console output (file always logs DEBUG).
    """
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    # Clear any existing handlers (important when running multiple sims, e.g. tests)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)
    rank = MPI.COMM_WORLD.rank

    if rank != 0:
        return

    # Console handler — clean output matching previous print() style
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    # File handler — detailed with timestamps
    file_handler = logging.FileHandler(output_dir / _LOG_FILE_NAME, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)


def teardown_logging() -> None:
    """Flush and close all handlers on the plm_data logger."""
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
