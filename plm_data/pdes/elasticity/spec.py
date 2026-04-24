"""Elasticity PDE spec."""

from plm_data.pdes.elasticity.pde import ElasticityPDE

PDE_SPEC = ElasticityPDE().spec

__all__ = ["PDE_SPEC"]
