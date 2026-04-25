"""Tests for plm_data.fields.source_terms."""

import pytest
import ufl
from dolfinx import fem

from plm_data.core.runtime_config import DomainConfig, FieldExpressionConfig
from plm_data.domains import create_domain
from plm_data.fields.source_terms import build_source_form, build_vector_source_form


@pytest.fixture
def mesh_2d():
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [4, 4]},
    )
    return create_domain(domain).mesh


@pytest.fixture
def test_function(mesh_2d):
    V = fem.functionspace(mesh_2d, ("Lagrange", 1))
    return ufl.TestFunction(V)


@pytest.fixture
def vector_test_function(mesh_2d):
    V = fem.functionspace(mesh_2d, ("Lagrange", 1, (2,)))
    return ufl.TestFunction(V)


def test_source_none(test_function, mesh_2d):
    config = FieldExpressionConfig(type="none", params={})
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is None


def test_source_custom(test_function, mesh_2d):
    config = FieldExpressionConfig(type="custom", params={})
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is None


def test_source_constant(test_function, mesh_2d):
    config = FieldExpressionConfig(type="constant", params={"value": 5.0})
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is not None


def test_source_sine_waves(test_function, mesh_2d):
    config = FieldExpressionConfig(
        type="sine_waves",
        params={
            "background": 0.0,
            "modes": [{"amplitude": 1.0, "cycles": [1, 1], "phase": 0.0}],
        },
    )
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is not None


def test_source_gaussian_bump(test_function, mesh_2d):
    config = FieldExpressionConfig(
        type="gaussian_bump",
        params={"amplitude": 2.0, "sigma": 0.1, "center": [0.5, 0.5]},
    )
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is not None


def test_source_with_param_references(test_function, mesh_2d):
    config = FieldExpressionConfig(
        type="constant",
        params={"value": "param:f_amplitude"},
    )
    parameters = {"f_amplitude": 3.14}
    result = build_source_form(test_function, mesh_2d, config, parameters)
    assert result is not None


def test_vector_source_form(vector_test_function, mesh_2d):
    config = FieldExpressionConfig(
        components={
            "x": FieldExpressionConfig(type="constant", params={"value": 1.0}),
            "y": FieldExpressionConfig(type="zero", params={}),
        }
    )

    result = build_vector_source_form(
        vector_test_function,
        mesh_2d,
        config,
        parameters={},
    )

    assert result is not None
