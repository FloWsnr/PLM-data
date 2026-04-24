"""Shallow-water PDE spec."""

from plm_data.pdes.shallow_water.pde import ShallowWaterPDE

PDE_SPEC = ShallowWaterPDE().spec

__all__ = ["PDE_SPEC"]
