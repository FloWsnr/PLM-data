"""Tests for plm_data.core.initial_conditions."""

import numpy as np
import pytest
from dolfinx import fem

from plm_data.core.config import ICConfig
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_domain


def _make_function(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    return fem.Function(V)


def test_apply_gaussian_bump(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = ICConfig(
        type="gaussian_bump",
        params={"sigma": 0.1, "amplitude": 1.0, "cx": 0.5, "cy": 0.5},
    )
    apply_ic(f, ic)
    assert np.max(f.x.array) > 0.5
    assert np.min(f.x.array) >= 0.0


def test_apply_constant(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = ICConfig(type="constant", params={"value": 3.14})
    apply_ic(f, ic)
    np.testing.assert_allclose(f.x.array, 3.14)


def test_unknown_ic_raises(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = ICConfig(type="nonexistent", params={})
    with pytest.raises(ValueError, match="Unknown initial condition"):
        apply_ic(f, ic)
