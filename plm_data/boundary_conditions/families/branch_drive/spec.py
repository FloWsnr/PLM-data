"""Branch-drive boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="branch_drive",
        description="Drive flow or scalar transport through a branched channel.",
        operators=("dirichlet", "neumann"),
        required_domain_roles=("inlet", "outlets", "walls"),
        supported_dimensions=(2, 3),
    )
)
