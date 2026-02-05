"""Output subsystem (frames, videos, trajectories, and metadata).

This package contains the concrete output handlers (png/gif/mp4/numpy/h5),
the OutputManager orchestration layer, and metadata helpers.

`pde_sim.core.output` re-exports the public API for backward compatibility.
"""

from .handlers import (
    GIFHandler,
    H5Handler,
    MP4Handler,
    NumpyHandler,
    Output1DHandler,
    OutputHandler,
    PNGHandler,
    create_output_handler,
)
from .manager import OutputManager
from .metadata import create_metadata

__all__ = [
    "OutputHandler",
    "Output1DHandler",
    "PNGHandler",
    "MP4Handler",
    "GIFHandler",
    "NumpyHandler",
    "H5Handler",
    "create_output_handler",
    "OutputManager",
    "create_metadata",
]

