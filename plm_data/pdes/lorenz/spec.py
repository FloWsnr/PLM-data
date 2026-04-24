"""Lorenz PDE spec."""

from plm_data.pdes.lorenz.pde import LorenzPDE

PDE_SPEC = LorenzPDE().spec

__all__ = ["PDE_SPEC"]
