"""OpenFOAM-backed runtime problems for fluid presets."""

from .compressible import OpenFOAMCompressibleNavierStokesProblem
from .incompressible import (
    OpenFOAMNavierStokesProblem,
    OpenFOAMThermalConvectionProblem,
)

__all__ = [
    "OpenFOAMCompressibleNavierStokesProblem",
    "OpenFOAMNavierStokesProblem",
    "OpenFOAMThermalConvectionProblem",
]
