"""Tests for plm_data.core.initial_conditions."""

from typing import cast

import numpy as np
import pytest
from dolfinx import fem

from plm_data.core.config import FieldExpressionConfig
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_domain


def _make_function(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    return cast(fem.Function, fem.Function(V))


def test_apply_gaussian_bump(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="gaussian_bump",
        params={"sigma": 0.1, "amplitude": 1.0, "center": [0.5, 0.5]},
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) > 0.5
    assert np.min(f.x.array) >= 0.0


def test_apply_constant(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(type="constant", params={"value": 3.14})
    apply_ic(f, ic, {})
    np.testing.assert_allclose(f.x.array, 3.14)


def test_apply_random_perturbation(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="random_perturbation",
        params={"mean": 0.5, "std": 0.01},
    )
    apply_ic(f, ic, {}, seed=42)
    assert np.mean(f.x.array) == pytest.approx(0.5, abs=0.05)
    assert np.std(f.x.array) > 0


def test_apply_sine_product(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="sine_product",
        params={"amplitude": 1.0, "kx": 1.0, "ky": 1.0},
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) > 0


def test_apply_step(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="step",
        params={"value_left": 1.0, "value_right": 0.0, "x_split": 0.5, "axis": 0},
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) == pytest.approx(1.0)
    assert np.min(f.x.array) == pytest.approx(0.0)


def test_apply_ic_with_param_refs(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(type="constant", params={"value": "param:initial_value"})
    apply_ic(f, ic, {"initial_value": 2.71})
    np.testing.assert_allclose(f.x.array, 2.71)


def test_custom_ic_is_noop(rectangle_domain):
    f = _make_function(rectangle_domain)
    f.x.array[:] = 99.0
    ic = FieldExpressionConfig(type="custom", params={})
    apply_ic(f, ic, {})
    np.testing.assert_allclose(f.x.array, 99.0)


def test_unknown_ic_raises(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(type="nonexistent", params={})
    with pytest.raises(ValueError, match="Unknown field type"):
        apply_ic(f, ic, {})
