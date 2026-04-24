"""Darcy PDE spec."""

from plm_data.pdes.darcy.pde import DarcyPDE

PDE_SPEC = DarcyPDE().spec

__all__ = ["PDE_SPEC"]
