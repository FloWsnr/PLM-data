"""Lid-driven cavity boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="lid_driven_cavity",
        description="Use no-slip walls with one driven lid on a Cartesian cavity.",
        operators=("dirichlet",),
        required_domain_roles=("walls",),
        required_any_domain_roles=(("y_max", "z_max"),),
        supported_dimensions=(2, 3),
        supported_field_shapes=("vector",),
    )
)
