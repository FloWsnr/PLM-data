"""Schrodinger PDE spec."""

from plm_data.pdes.schrodinger.pde import SchrodingerPDE

PDE_SPEC = SchrodingerPDE().spec

__all__ = ["PDE_SPEC"]
