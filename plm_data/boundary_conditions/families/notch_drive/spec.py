"""Notch-drive boundary-condition family spec."""

from plm_data.boundary_conditions.base import (
    BoundaryFamilySpec,
    register_boundary_family_spec,
)

FAMILY_SPEC = register_boundary_family_spec(
    BoundaryFamilySpec(
        name="notch_drive",
        description="Use different boundary behavior on outer and notch boundaries.",
        operators=("dirichlet", "neumann", "robin"),
        required_domain_roles=("outer", "notch"),
        supported_dimensions=(2,),
    )
)
