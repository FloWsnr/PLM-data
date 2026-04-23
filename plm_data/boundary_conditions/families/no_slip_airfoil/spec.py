"""No-slip airfoil boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="no_slip_airfoil",
        description="Use vector no-slip conditions on solid walls and an airfoil.",
        operators=("dirichlet",),
        required_domain_roles=("solid", "airfoil"),
        supported_dimensions=(2, 3),
        supported_field_shapes=("vector",),
    )
)
