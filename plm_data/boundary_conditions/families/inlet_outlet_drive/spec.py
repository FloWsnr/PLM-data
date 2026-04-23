"""Inlet/outlet drive boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="inlet_outlet_drive",
        description="Drive a channel through inlet and outlet boundary roles.",
        operators=("dirichlet", "neumann"),
        required_domain_roles=("inlet",),
        required_any_domain_roles=(("outlet", "outlets"),),
        supported_dimensions=(2, 3),
    )
)
