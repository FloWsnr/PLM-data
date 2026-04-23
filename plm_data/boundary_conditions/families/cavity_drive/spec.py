"""Side-cavity drive boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="cavity_drive",
        description="Use wall-driven behavior on a channel with a side cavity.",
        operators=("dirichlet", "neumann"),
        required_domain_roles=("open", "solid", "walls"),
        supported_dimensions=(2, 3),
    )
)
