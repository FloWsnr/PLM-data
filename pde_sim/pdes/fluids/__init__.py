"""Fluid Dynamics PDEs: Navier-Stokes, Shallow Water, etc."""

from .vorticity import VorticityPDE
from .shallow_water import ShallowWaterPDE
from .navier_stokes import NavierStokesPDE
from .thermal_convection import ThermalConvectionPDE
from .method_of_images import MethodOfImagesPDE
from .dipoles import DipolesPDE

__all__ = [
    "VorticityPDE",
    "ShallowWaterPDE",
    "NavierStokesPDE",
    "ThermalConvectionPDE",
    "MethodOfImagesPDE",
    "DipolesPDE",
]
