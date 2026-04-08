"""Centralized logging setup for PLM-data simulations."""

import logging
import sys
from pathlib import Path

from mpi4py import MPI

_PACKAGE_LOGGER_NAME = "plm_data"
_LOG_FILE_NAME = "simulation.log"


def _reactivate_package_loggers() -> None:
    logger_dict = logging.root.manager.loggerDict
    for name, logger_obj in logger_dict.items():
        if not isinstance(logger_obj, logging.Logger):
            continue
        if name == _PACKAGE_LOGGER_NAME or name.startswith(f"{_PACKAGE_LOGGER_NAME}."):
            logger_obj.disabled = False


def _clear_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _build_console_handler(level: int) -> logging.Handler:
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))
    return console


def _build_file_handler(path: Path, *, mode: str, level: int) -> logging.Handler:
    file_handler = logging.FileHandler(path, mode=mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    return file_handler


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the plm_data namespace."""
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def _configure_package_logging(
    output_dir: Path, *, console_level: int, file_mode: str
) -> None:
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    _reactivate_package_loggers()
    _clear_handlers(logging.getLogger())

    # Clear any existing handlers (important when running multiple sims, e.g. tests)
    _clear_handlers(logger)

    logger.disabled = False
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    rank = MPI.COMM_WORLD.rank

    if rank != 0:
        return

    # Console handler — clean output matching previous print() style
    logger.addHandler(_build_console_handler(console_level))

    # File handler — detailed with timestamps
    logger.addHandler(
        _build_file_handler(
            output_dir / _LOG_FILE_NAME,
            mode=file_mode,
            level=logging.DEBUG,
        )
    )


def setup_logging(output_dir: Path, console_level: int = logging.INFO) -> None:
    """Configure file and console logging for a simulation run.

    Args:
        output_dir: Directory for the log file (simulation.log).
        console_level: Logging level for console output (file always logs DEBUG).
    """
    _configure_package_logging(
        output_dir,
        console_level=console_level,
        file_mode="w",
    )


def teardown_logging() -> None:
    """Flush and close all handlers on the plm_data logger."""
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    _clear_handlers(logger)
    _clear_handlers(logging.getLogger())
