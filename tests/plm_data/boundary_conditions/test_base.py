"""Tests for first-class boundary-condition specifications."""

import pytest

from plm_data.boundary_conditions import (
    MAXWELL_BOUNDARY_OPERATORS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
    get_boundary_operator_spec,
    list_boundary_operator_specs,
)
import plm_data.boundary_conditions.base as boundary_base
from plm_data.boundary_conditions.base import (
    BoundaryOperatorParameterSpec,
    BoundaryOperatorSpec,
    register_boundary_operator_spec,
)
from plm_data.pdes import metadata as pde_metadata


def test_register_boundary_operator_spec_adds_spec(monkeypatch):
    registry: dict[str, BoundaryOperatorSpec] = {}
    monkeypatch.setattr(boundary_base, "_BOUNDARY_OPERATOR_REGISTRY", registry)
    spec = BoundaryOperatorSpec(name="unit_test", value_shape="field")

    registered = register_boundary_operator_spec(spec)

    assert registered is spec
    assert registry == {"unit_test": spec}


def test_boundary_operator_parameter_spec_rejects_invalid_sampling_bounds():
    with pytest.raises(ValueError, match="both sampling_min and sampling_max"):
        BoundaryOperatorParameterSpec(
            name="bad",
            kind="float",
            sampling_min=0.0,
        )

    with pytest.raises(ValueError, match="sampling_min greater than sampling_max"):
        BoundaryOperatorParameterSpec(
            name="bad",
            kind="float",
            sampling_min=2.0,
            sampling_max=1.0,
        )

    with pytest.raises(ValueError, match="sampling_min must be >= hard_min"):
        BoundaryOperatorParameterSpec(
            name="bad",
            kind="float",
            hard_min=0.5,
            sampling_min=0.2,
            sampling_max=1.0,
        )


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


def test_pde_metadata_reexports_boundary_operator_specs():
    assert (
        pde_metadata.SCALAR_STANDARD_BOUNDARY_OPERATORS
        is SCALAR_STANDARD_BOUNDARY_OPERATORS
    )
    assert (
        pde_metadata.VECTOR_STANDARD_BOUNDARY_OPERATORS
        is VECTOR_STANDARD_BOUNDARY_OPERATORS
    )
    assert pde_metadata.MAXWELL_BOUNDARY_OPERATORS is MAXWELL_BOUNDARY_OPERATORS
