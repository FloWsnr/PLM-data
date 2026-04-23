"""All-Dirichlet boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="all_dirichlet",
        description="Apply one Dirichlet operator to every domain boundary.",
        operators=("dirichlet",),
        required_domain_roles=("all",),
    )
)
