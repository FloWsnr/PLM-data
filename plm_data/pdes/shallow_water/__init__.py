"""Shallow-water PDE package."""

from plm_data.pdes.shallow_water.pde import ShallowWaterPDE
from plm_data.pdes.shallow_water.spec import PDE_SPEC

__all__ = ["PDE_SPEC", "ShallowWaterPDE"]
