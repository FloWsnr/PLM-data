"""Axis-periodic boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="periodic_axis",
        description="Make one available domain side pair periodic.",
        operators=("periodic",),
        required_any_domain_roles=(("x_pair", "y_pair", "z_pair"),),
        requires_periodic_pairs=True,
    )
)
