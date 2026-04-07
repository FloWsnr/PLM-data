"""Centralized logging setup for PLM-data simulations."""

import logging
import sys
from pathlib import Path

from mpi4py import MPI

_PACKAGE_LOGGER_NAME = "plm_data"
_LOG_FILE_NAME = "simulation.log"
_CURRENT_OUTPUT_DIR: Path | None = None
_CURRENT_CONSOLE_LEVEL: int = logging.INFO
_CLAWPACK_CHILD_LOGGERS = (
    "pyclaw.controller",
    "pyclaw.solver",
    "pyclaw.solution",
    "pyclaw.fileio",
)
_CLAWPACK_TOP_LEVEL_LOGGERS = ("pyclaw", "data", "f2py")


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


def _setup_external_root_logging() -> None:
    root_logger = logging.getLogger()
    _clear_handlers(root_logger)
    root_logger.disabled = False
    root_logger.setLevel(logging.WARNING)
    if _CURRENT_OUTPUT_DIR is None or MPI.COMM_WORLD.rank != 0:
        return
    root_logger.addHandler(_build_console_handler(logging.WARNING))
    root_logger.addHandler(
        _build_file_handler(
            _CURRENT_OUTPUT_DIR / _LOG_FILE_NAME,
            mode="a",
            level=logging.WARNING,
        )
    )


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the plm_data namespace."""
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def setup_logging(output_dir: Path, console_level: int = logging.INFO) -> None:
    """Configure file and console logging for a simulation run.

    Args:
        output_dir: Directory for the log file (simulation.log).
        console_level: Logging level for console output (file always logs DEBUG).
    """
    global _CURRENT_OUTPUT_DIR, _CURRENT_CONSOLE_LEVEL

    _CURRENT_OUTPUT_DIR = output_dir
    _CURRENT_CONSOLE_LEVEL = console_level
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
        _build_file_handler(output_dir / _LOG_FILE_NAME, mode="w", level=logging.DEBUG)
    )


def restore_logging() -> None:
    """Reinstall PLM-data handlers after third-party logging reconfiguration."""

    if _CURRENT_OUTPUT_DIR is None:
        return
    setup_logging(_CURRENT_OUTPUT_DIR, console_level=_CURRENT_CONSOLE_LEVEL)


def configure_clawpack_logging() -> None:
    """Route Clawpack warnings/errors into PLM-data logging without CLAW handlers."""

    restore_logging()
    _setup_external_root_logging()

    for name in (*_CLAWPACK_TOP_LEVEL_LOGGERS, *_CLAWPACK_CHILD_LOGGERS):
        logger = logging.getLogger(name)
        _clear_handlers(logger)
        logger.disabled = False
        logger.propagate = True

    logging.getLogger("pyclaw").setLevel(logging.WARNING)
    for name in _CLAWPACK_CHILD_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    for name in ("data", "f2py"):
        logging.getLogger(name).setLevel(logging.WARNING)


def teardown_logging() -> None:
    """Flush and close all handlers on the plm_data logger."""
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    _clear_handlers(logger)
    _clear_handlers(logging.getLogger())
