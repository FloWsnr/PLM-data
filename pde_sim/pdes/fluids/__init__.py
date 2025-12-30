"""Fluid Dynamics PDEs: Navier-Stokes, Shallow Water, etc."""

from .vorticity import VorticityPDE, ShallowWaterPDE

__all__ = [
    "VorticityPDE",
    "ShallowWaterPDE",
]
