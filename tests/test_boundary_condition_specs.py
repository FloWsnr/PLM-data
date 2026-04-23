"""Tests for first-class boundary-condition specifications."""

from plm_data.boundary_conditions import (
    MAXWELL_BOUNDARY_OPERATORS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
    get_boundary_family_spec,
    get_boundary_operator_spec,
    list_boundary_family_specs,
    list_boundary_operator_specs,
)
from plm_data.domains import list_domain_specs
from plm_data.presets import metadata as preset_metadata


def test_boundary_operator_registry_covers_standard_operators():
    operators = list_boundary_operator_specs()

    assert set(operators) == {
        "absorbing",
        "dirichlet",
        "neumann",
        "periodic",
        "robin",
        "simply_supported",
    }
    assert set(SCALAR_STANDARD_BOUNDARY_OPERATORS) == {
        "dirichlet",
        "neumann",
        "robin",
        "periodic",
    }
    assert set(VECTOR_STANDARD_BOUNDARY_OPERATORS) == {
        "dirichlet",
        "neumann",
        "periodic",
    }
    assert set(MAXWELL_BOUNDARY_OPERATORS) == {
        "dirichlet",
        "periodic",
        "absorbing",
    }

    robin = get_boundary_operator_spec("robin")
    assert robin.allowed_field_shapes == ("scalar",)
    assert robin.operator_parameter_names == ("alpha",)
    assert set(robin.parameter_specs) == {"alpha"}
    assert robin.parameter_specs["alpha"].sampling_min == 0.0

    periodic = get_boundary_operator_spec("periodic")
    assert periodic.value_shape is None
    assert periodic.requires_pair_with is True


def test_preset_metadata_reexports_boundary_operator_specs():
    assert (
        preset_metadata.SCALAR_STANDARD_BOUNDARY_OPERATORS
        is SCALAR_STANDARD_BOUNDARY_OPERATORS
    )
    assert (
        preset_metadata.VECTOR_STANDARD_BOUNDARY_OPERATORS
        is VECTOR_STANDARD_BOUNDARY_OPERATORS
    )
    assert preset_metadata.MAXWELL_BOUNDARY_OPERATORS is MAXWELL_BOUNDARY_OPERATORS


def test_boundary_family_registry_covers_domain_recipes():
    families = list_boundary_family_specs()

    assert set(families) == {
        "all_dirichlet",
        "all_neumann",
        "all_robin",
        "branch_drive",
        "cavity_drive",
        "full_periodic",
        "inlet_outlet_drive",
        "inner_outer_drive",
        "lid_driven_cavity",
        "no_slip_airfoil",
        "no_slip_obstacle",
        "notch_drive",
        "open_channel",
        "outer_hole_drive",
        "periodic_axis",
        "porous_obstacle_drive",
    }

    operators = set(list_boundary_operator_specs())
    for family in families.values():
        assert set(family.operators).issubset(operators)


def test_domain_allowed_boundary_families_are_compatible():
    for domain_spec in list_domain_specs().values():
        assert domain_spec.allowed_boundary_families
        for family_name in domain_spec.allowed_boundary_families:
            family = get_boundary_family_spec(family_name)
            assert family.is_compatible_with_domain(domain_spec)
