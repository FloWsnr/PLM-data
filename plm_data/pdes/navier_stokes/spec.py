"""Navier-Stokes PDE spec."""

from plm_data.pdes.navier_stokes.pde import NavierStokesPDE

PDE_SPEC = NavierStokesPDE().spec

__all__ = ["PDE_SPEC"]
