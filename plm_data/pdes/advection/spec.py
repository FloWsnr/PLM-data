"""Advection PDE spec."""

from plm_data.pdes.advection.pde import AdvectionPDE

PDE_SPEC = AdvectionPDE().spec

__all__ = ["PDE_SPEC"]
