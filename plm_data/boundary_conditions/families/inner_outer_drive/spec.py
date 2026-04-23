"""Inner/outer drive boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="inner_outer_drive",
        description="Use distinct values or fluxes on annulus inner and outer walls.",
        operators=("dirichlet", "neumann", "robin"),
        required_domain_roles=("inner", "outer"),
        supported_dimensions=(2, 3),
    )
)
