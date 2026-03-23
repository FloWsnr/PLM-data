"""Tests for plm_data.core.interpolation."""

import numpy as np
from dolfinx import fem

from plm_data.core.interpolation import function_to_array
from plm_data.core.mesh import create_domain


def test_function_to_array(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    f = fem.Function(V)
    f.x.array[:] = 1.0  # type: ignore[reportAttributeAccessIssue]

    arr = function_to_array(f, resolution=(4, 4))  # type: ignore[reportArgumentType]
    assert arr.shape == (4, 4)
    np.testing.assert_allclose(arr, 1.0, atol=1e-10)
