"""Open-channel boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="open_channel",
        description="Use open inlet/outlet boundaries and solid wall boundaries.",
        operators=("dirichlet", "neumann"),
        required_domain_roles=("open", "solid"),
        supported_dimensions=(2, 3),
    )
)
