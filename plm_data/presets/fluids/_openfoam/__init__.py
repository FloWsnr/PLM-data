"""OpenFOAM-backed runtime problems for fluid presets."""

from .compressible import OpenFOAMCompressibleNavierStokesProblem
from .euler import OpenFOAMEulerProblem
from .incompressible import (
    OpenFOAMNavierStokesProblem,
    OpenFOAMThermalConvectionProblem,
)
from .mhd import OpenFOAMMHDProblem

__all__ = [
    "OpenFOAMCompressibleNavierStokesProblem",
    "OpenFOAMEulerProblem",
    "OpenFOAMMHDProblem",
    "OpenFOAMNavierStokesProblem",
    "OpenFOAMThermalConvectionProblem",
]
