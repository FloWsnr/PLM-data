"""Tests for plm_data.core.mesh."""

import importlib.util
import sys
import types

import numpy as np
import pytest
from mpi4py import MPI

import plm_data.core.mesh as mesh_module
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


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (
            {"inner_radius": 0.0, "outer_radius": 1.0, "mesh_size": 0.2},
            "inner_radius",
        ),
        (
            {"inner_radius": 0.3, "outer_radius": -1.0, "mesh_size": 0.2},
            "outer_radius",
        ),
        (
            {"inner_radius": 1.0, "outer_radius": 1.0, "mesh_size": 0.2},
            "inner_radius' < 'outer_radius",
        ),
        (
            {"inner_radius": 0.3, "outer_radius": 1.0, "mesh_size": 0.0},
            "mesh_size",
        ),
    ],
)
def test_annulus_parameter_validation(params, match):
    with pytest.raises(ValueError, match=match):
        create_domain(DomainConfig(type="annulus", params=params))


def test_model_to_mesh_shared_facet_uses_partitioner(monkeypatch):
    calls: dict[str, object] = {}
    sentinel_partitioner = object()

    def _fake_create_cell_partitioner(ghost_mode, max_links):
        calls["ghost_mode"] = ghost_mode
        calls["max_links"] = max_links
        return sentinel_partitioner

    def _fake_model_to_mesh(model, comm, *, rank, gdim, partitioner):
        calls["model"] = model
        calls["comm"] = comm
        calls["rank"] = rank
        calls["gdim"] = gdim
        calls["partitioner"] = partitioner
        return "mesh-data"

    monkeypatch.setattr(
        mesh_module.mesh,
        "create_cell_partitioner",
        _fake_create_cell_partitioner,
    )
    monkeypatch.setitem(
        sys.modules,
        "dolfinx.io.gmsh",
        types.SimpleNamespace(model_to_mesh=_fake_model_to_mesh),
    )

    result = mesh_module._model_to_mesh_shared_facet(
        np.array([[0.0]]),
        MPI.COMM_WORLD,
        rank=0,
        gdim=2,
    )

    assert result == "mesh-data"
    assert calls["ghost_mode"] == mesh_module.GhostMode.shared_facet
    assert calls["max_links"] == 2
    assert calls["partitioner"] is sentinel_partitioner
