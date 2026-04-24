"""Kuramoto-Sivashinsky PDE spec."""

from plm_data.pdes.kuramoto_sivashinsky.pde import KuramotoSivashinskyPDE

PDE_SPEC = KuramotoSivashinskyPDE().spec

__all__ = ["PDE_SPEC"]
