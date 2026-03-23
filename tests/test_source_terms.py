"""Tests for plm_data.core.source_terms."""

import ufl
import pytest
from dolfinx import fem

from plm_data.core.config import DomainConfig, SourceTermConfig
from plm_data.core.mesh import create_domain
from plm_data.core.source_terms import build_source_form


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


def test_source_none(test_function, mesh_2d):
    """type='none' returns None."""
    config = SourceTermConfig(type="none", params={})
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is None


def test_source_custom(test_function, mesh_2d):
    """type='custom' returns None (preset handles it)."""
    config = SourceTermConfig(type="custom", params={})
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is None


def test_source_constant(test_function, mesh_2d):
    """type='constant' returns a UFL form."""
    config = SourceTermConfig(type="constant", params={"value": 5.0})
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is not None


def test_source_sine_product(test_function, mesh_2d):
    """type='sine_product' returns a UFL form."""
    config = SourceTermConfig(
        type="sine_product",
        params={"amplitude": 1.0, "kx": 1, "ky": 1},
    )
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is not None


def test_source_gaussian_bump(test_function, mesh_2d):
    """type='gaussian_bump' returns a UFL form."""
    config = SourceTermConfig(
        type="gaussian_bump",
        params={"amplitude": 2.0, "sigma": 0.1, "center": [0.5, 0.5]},
    )
    result = build_source_form(test_function, mesh_2d, config, parameters={})
    assert result is not None


def test_source_with_param_references(test_function, mesh_2d):
    """Parameter references in params are resolved from the parameters dict."""
    config = SourceTermConfig(
        type="constant",
        params={"value": "param:f_amplitude"},
    )
    parameters = {"f_amplitude": 3.14}
    result = build_source_form(test_function, mesh_2d, config, parameters)
    assert result is not None
