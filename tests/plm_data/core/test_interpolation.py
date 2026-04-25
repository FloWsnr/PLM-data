"""Tests for plm_data.core.interpolation."""

import importlib.util

import numpy as np
import pytest
from dolfinx import fem

from plm_data.core.runtime_config import DomainConfig
from plm_data.core.interpolation import function_to_array, function_to_grid
from plm_data.domains import create_domain

HAS_GMSH = importlib.util.find_spec("gmsh") is not None


def test_function_to_array(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    f = fem.Function(V)
    f.x.array[:] = 1.0  # type: ignore[reportAttributeAccessIssue]

    arr, _ = function_to_array(f, resolution=(4, 4))  # type: ignore[reportArgumentType]
    assert arr.shape == (4, 4)
    np.testing.assert_allclose(arr, 1.0, atol=1e-10)


def test_function_to_grid_vector(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1, (2,)))
    f = fem.Function(V)
    f.interpolate(lambda x: np.vstack((x[0], x[1])))

    arr, _ = function_to_grid(f, resolution=(4, 4))  # type: ignore[reportArgumentType]
    assert arr.shape == (2, 4, 4)
    np.testing.assert_allclose(arr[0, :, 0], np.linspace(0.0, 1.0, 4), atol=1e-10)
    np.testing.assert_allclose(arr[1, 0, :], np.linspace(0.0, 1.0, 4), atol=1e-10)


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_function_to_array_annulus():
    domain_geom = create_domain(
        DomainConfig(
            type="annulus",
            params={
                "center": [0.2, -0.1],
                "inner_radius": 0.3,
                "outer_radius": 1.0,
                "mesh_size": 0.15,
            },
        )
    )
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    f = fem.Function(V)
    f.x.array[:] = 1.0  # type: ignore[reportAttributeAccessIssue]

    arr, cache = function_to_array(f, resolution=(8, 8))  # type: ignore[reportArgumentType]
    assert arr.shape == (8, 8)
    assert np.isnan(arr).any(), "Annulus grid should have NaN outside the domain"
    valid = ~np.isnan(arr)
    np.testing.assert_allclose(arr[valid], 1.0, atol=1e-10)
    assert cache.outside_mask is not None
    assert cache.outside_mask.sum() > 0
