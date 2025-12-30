"""Fluid Dynamics PDEs: Navier-Stokes, Shallow Water, etc."""

from .vorticity import VorticityPDE, ShallowWaterPDE
from .advanced import (
    NavierStokesPDE,
    ThermalConvectionPDE,
    MethodOfImagesPDE,
    DipolesPDE,
)

__all__ = [
    "VorticityPDE",
    "ShallowWaterPDE",
    "NavierStokesPDE",
    "ThermalConvectionPDE",
    "MethodOfImagesPDE",
    "DipolesPDE",
]
