"""Tests for plm_data.core.mesh."""

from plm_data.core.mesh import create_domain


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
