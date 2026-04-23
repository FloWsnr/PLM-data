"""No-slip obstacle boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="no_slip_obstacle",
        description="Use vector no-slip conditions on solid walls and an obstacle.",
        operators=("dirichlet",),
        required_domain_roles=("solid", "obstacle"),
        supported_dimensions=(2, 3),
        supported_field_shapes=("vector",),
    )
)
