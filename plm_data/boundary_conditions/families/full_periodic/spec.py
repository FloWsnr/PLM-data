"""Fully periodic boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="full_periodic",
        description="Make all available periodic domain side pairs periodic.",
        operators=("periodic",),
        requires_periodic_pairs=True,
    )
)
