"""OpenFOAM-backed runtime problems for fluid presets."""

from .compressible import OpenFOAMCompressibleNavierStokesProblem
from .incompressible import (
    OpenFOAMNavierStokesProblem,
    OpenFOAMThermalConvectionProblem,
)
from .mhd import OpenFOAMMHDProblem

__all__ = [
    "OpenFOAMCompressibleNavierStokesProblem",
    "OpenFOAMMHDProblem",
    "OpenFOAMNavierStokesProblem",
    "OpenFOAMThermalConvectionProblem",
]
