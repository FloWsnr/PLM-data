"""Schnakenberg PDE spec."""

from plm_data.pdes.schnakenberg.pde import SchnakenbergPDE

PDE_SPEC = SchnakenbergPDE().spec

__all__ = ["PDE_SPEC"]
