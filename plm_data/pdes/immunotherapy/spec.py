"""Immunotherapy PDE spec."""

from plm_data.pdes.immunotherapy.pde import ImmunotherapyPDE

PDE_SPEC = ImmunotherapyPDE().spec

__all__ = ["PDE_SPEC"]
