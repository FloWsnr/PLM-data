"""Superlattice PDE spec."""

from plm_data.pdes.superlattice.pde import SuperlatticePDE

PDE_SPEC = SuperlatticePDE().spec

__all__ = ["PDE_SPEC"]
