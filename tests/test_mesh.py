"""Tests for plm_data.core.mesh."""

from plm_data.core.mesh import create_mesh


def test_create_rectangle_mesh(rectangle_domain):
    msh = create_mesh(rectangle_domain)
    assert msh.topology.dim == 2
    assert msh.geometry.dim == 2
    num_cells = msh.topology.index_map(2).size_global
    assert num_cells > 0
