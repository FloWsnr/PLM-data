"""Tests for plm_data.pdes.boundary_validation."""

from types import SimpleNamespace

from plm_data.core.runtime_config import BoundaryConditionConfig
from plm_data.pdes.boundary_validation import validate_vector_standard_boundary_field
from tests.runtime_helpers import boundary_field_config, vector_zero


def test_validate_vector_standard_boundary_field_accepts_default_operators():
    domain_geom = SimpleNamespace(boundary_names={"x-": 1, "x+": 2})
    boundary_field = boundary_field_config(
        {
            "x-": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
            "x+": BoundaryConditionConfig(type="neumann", value=vector_zero()),
        }
    )

    validate_vector_standard_boundary_field(
        pde_name="elasticity",
        field_name="u",
        boundary_field=boundary_field,
        domain_geom=domain_geom,
    )
