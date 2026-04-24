"""Plate PDE spec."""

from plm_data.pdes.plate.pde import PlatePDE

PDE_SPEC = PlatePDE().spec

__all__ = ["PDE_SPEC"]
