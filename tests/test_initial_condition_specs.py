"""Tests for first-class initial-condition specifications."""

import pytest

from plm_data.core.config import (
    FieldExpressionConfig,
    _validate_initial_condition_scalar_expression,
)
from plm_data.domains import list_domain_specs
from plm_data.initial_conditions import (
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    get_initial_condition_spec,
    list_initial_condition_specs,
)


def test_initial_condition_registry_covers_config_families():
    specs = list_initial_condition_specs()

    assert set(specs) == {
        "affine",
        "constant",
        "custom",
        "gaussian_blobs",
        "gaussian_bump",
        "gaussian_noise",
        "gaussian_wave_packet",
        "none",
        "quadrants",
        "radial_cosine",
        "sine_waves",
        "step",
        "zero",
    }
    assert set(COMMON_SCALAR_INITIAL_CONDITION_FAMILIES) == {
        "constant",
        "gaussian_blobs",
        "gaussian_bump",
        "gaussian_noise",
        "gaussian_wave_packet",
        "quadrants",
        "radial_cosine",
        "sine_waves",
        "step",
    }
    assert {name for name, spec in specs.items() if spec.common_scalar_family} == set(
        COMMON_SCALAR_INITIAL_CONDITION_FAMILIES
    )


def test_initial_condition_specs_expose_constraints():
    gaussian_noise = get_initial_condition_spec("gaussian_noise")
    assert gaussian_noise.requires_seed is True
    assert set(gaussian_noise.parameters) == {"mean", "std"}
    assert gaussian_noise.parameters["std"].hard_min == 0.0

    gaussian_blobs = get_initial_condition_spec("gaussian_blobs")
    assert gaussian_blobs.requires_seed is True
    assert set(gaussian_blobs.parameters) == {"background", "generators"}

    affine = get_initial_condition_spec("affine")
    assert affine.parameters["x"].required is False
    assert affine.parameters["z"].required is False

    for name in ("none", "zero", "custom"):
        assert get_initial_condition_spec(name).supports_vector_field_level is True


def test_domain_allowed_initial_condition_families_are_compatible():
    for domain_spec in list_domain_specs().values():
        assert domain_spec.allowed_initial_condition_families
        for family_name in domain_spec.allowed_initial_condition_families:
            family = get_initial_condition_spec(family_name)
            assert family.is_compatible_with_domain(domain_spec)


def test_config_validation_uses_initial_condition_registry_for_allowed_types():
    with pytest.raises(ValueError, match="unsupported initial_condition type"):
        _validate_initial_condition_scalar_expression(
            FieldExpressionConfig(type="unknown_family", params={}),
            "inputs.u.initial_condition",
            gdim=2,
        )
