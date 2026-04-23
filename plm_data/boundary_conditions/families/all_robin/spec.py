"""All-Robin boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="all_robin",
        description="Apply one scalar Robin operator to every domain boundary.",
        operators=("robin",),
        required_domain_roles=("all",),
        supported_field_shapes=("scalar",),
    )
)
