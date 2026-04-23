"""Porous-obstacle drive boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="porous_obstacle_drive",
        description="Use separate behavior for open boundaries and porous obstacles.",
        operators=("dirichlet", "neumann"),
        required_domain_roles=("open", "solid", "obstacles"),
        supported_dimensions=(2, 3),
    )
)
