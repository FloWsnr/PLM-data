"""Brusselator PDE spec."""

from plm_data.pdes.brusselator.pde import BrusselatorPDE

PDE_SPEC = BrusselatorPDE().spec

__all__ = ["PDE_SPEC"]
