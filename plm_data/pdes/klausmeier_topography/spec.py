"""Klausmeier topography PDE spec."""

from plm_data.pdes.klausmeier_topography.pde import KlausmeierTopographyPDE

PDE_SPEC = KlausmeierTopographyPDE().spec

__all__ = ["PDE_SPEC"]
