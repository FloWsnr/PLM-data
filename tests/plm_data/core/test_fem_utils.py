"""Tests for plm_data.core.fem_utils."""

import numpy as np
import pytest
import ufl
from dolfinx import fem

from plm_data.core.fem_utils import dg_jump, domain_average
from plm_data.core.runtime_config import DomainConfig
from plm_data.domains import create_domain


@pytest.fixture
def mesh_2d():
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [4, 4]},
    )
    return create_domain(domain).mesh


def test_domain_average_of_constant_field(mesh_2d):
    V = fem.functionspace(mesh_2d, ("Lagrange", 1))
    field = fem.Function(V)
    field.interpolate(lambda x: np.full(x.shape[1], 3.5))

    assert domain_average(mesh_2d, field) == pytest.approx(3.5)


def test_dg_jump_builds_facet_tensor_expression(mesh_2d):
    V = fem.functionspace(mesh_2d, ("Lagrange", 1, (2,)))
    phi = ufl.TrialFunction(V)
    normal = ufl.FacetNormal(mesh_2d)

    result = dg_jump(phi, normal)

    assert result.ufl_shape == (2, 2)
