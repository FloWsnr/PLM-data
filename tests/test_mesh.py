"""Tests for plm_data.core.mesh."""

import importlib.util

import pytest

from plm_data.core.config import DomainConfig
from plm_data.core.mesh import create_domain

HAS_GMSH = importlib.util.find_spec("gmsh") is not None


def test_create_rectangle_mesh(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    msh = domain_geom.mesh
    assert msh.topology.dim == 2
    assert msh.geometry.dim == 2
    num_cells = msh.topology.index_map(2).size_global
    assert num_cells > 0


def test_rectangle_boundary_tags(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    assert set(domain_geom.boundary_names.keys()) == {"x-", "x+", "y-", "y+"}
    for name, tag in domain_geom.boundary_names.items():
        facets = domain_geom.facet_tags.find(tag)
        assert len(facets) > 0, f"No facets tagged for boundary '{name}'"


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_create_annulus_mesh():
    domain = DomainConfig(
        type="annulus",
        params={"inner_radius": 0.3, "outer_radius": 1.0, "mesh_size": 0.2},
    )
    domain_geom = create_domain(domain)
    msh = domain_geom.mesh
    assert msh.topology.dim == 2
    assert msh.geometry.dim == 2
    assert msh.topology.index_map(2).size_global > 0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_annulus_boundary_tags():
    domain = DomainConfig(
        type="annulus",
        params={"inner_radius": 0.3, "outer_radius": 1.0, "mesh_size": 0.2},
    )
    domain_geom = create_domain(domain)
    assert set(domain_geom.boundary_names.keys()) == {"inner", "outer"}
    for name, tag in domain_geom.boundary_names.items():
        facets = domain_geom.facet_tags.find(tag)
        assert len(facets) > 0, f"No facets tagged for boundary '{name}'"
