"""Backward-compatible re-exports for the output subsystem.

Historically, output logic lived in this module. It has been moved to
`pde_sim.output` to keep the `core` package focused on orchestration/config.
"""

from pde_sim.output.handlers import (
    GIFHandler,
    H5Handler,
    MP4Handler,
    NumpyHandler,
    Output1DHandler,
    Output3DHandler,
    OutputHandler,
    PNGHandler,
    create_output_handler,
)
from pde_sim.output.manager import OutputManager
from pde_sim.output.metadata import create_metadata

__all__ = [
    "OutputHandler",
    "Output1DHandler",
    "Output3DHandler",
    "PNGHandler",
    "MP4Handler",
    "GIFHandler",
    "NumpyHandler",
    "H5Handler",
    "create_output_handler",
    "OutputManager",
    "create_metadata",
]

