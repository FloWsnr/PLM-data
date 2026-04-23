"""All-Neumann boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="all_neumann",
        description="Apply one Neumann operator to every domain boundary.",
        operators=("neumann",),
        required_domain_roles=("all",),
    )
)
