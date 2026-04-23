"""Outer/hole drive boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="outer_hole_drive",
        description="Use different boundary behavior on plate outer and hole walls.",
        operators=("dirichlet", "neumann", "robin"),
        required_domain_roles=("outer", "holes"),
        supported_dimensions=(2,),
    )
)
