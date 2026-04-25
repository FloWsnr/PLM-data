"""Tests for first-class initial-condition operator specifications."""

from plm_data.initial_conditions import (
    COMMON_SCALAR_INITIAL_CONDITION_OPERATORS,
    get_initial_condition_operator_spec,
    has_initial_condition_operator_spec,
    list_initial_condition_operator_specs,
)
import plm_data.initial_conditions.base as ic_base
from plm_data.initial_conditions.base import (
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)


def test_register_initial_condition_operator_spec_adds_spec(monkeypatch):
    registry: dict[str, InitialConditionOperatorSpec] = {}
    monkeypatch.setattr(ic_base, "_INITIAL_CONDITION_OPERATOR_REGISTRY", registry)
    spec = InitialConditionOperatorSpec(
        name="unit_test",
        description="test operator",
        parameters={},
    )

    registered = register_initial_condition_operator_spec(spec)

    assert registered is spec
    assert registry == {"unit_test": spec}


def test_initial_condition_operator_registry_covers_runtime_operators():
    specs = list_initial_condition_operator_specs()

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
    assert set(COMMON_SCALAR_INITIAL_CONDITION_OPERATORS) == {
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
    assert {name for name, spec in specs.items() if spec.common_scalar_operator} == set(
        COMMON_SCALAR_INITIAL_CONDITION_OPERATORS
    )


def test_initial_condition_specs_expose_constraints():
    gaussian_noise = get_initial_condition_operator_spec("gaussian_noise")
    assert gaussian_noise.requires_seed is True
    assert set(gaussian_noise.parameters) == {"mean", "std"}
    assert gaussian_noise.parameters["std"].hard_min == 0.0

    gaussian_blobs = get_initial_condition_operator_spec("gaussian_blobs")
    assert gaussian_blobs.requires_seed is True
    assert set(gaussian_blobs.parameters) == {"background", "generators"}

    affine = get_initial_condition_operator_spec("affine")
    assert affine.parameters["x"].required is False
    assert affine.parameters["z"].required is False

    for name in ("none", "zero", "custom"):
        assert (
            get_initial_condition_operator_spec(name).supports_vector_field_level
            is True
        )


def test_initial_condition_registry_rejects_unknown_names():
    assert has_initial_condition_operator_spec("unknown_scenario") is False
