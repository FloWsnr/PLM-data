"""Tests for plm_data.core.mesh."""

import importlib.util
import sys
import types

import numpy as np
import pytest
from dolfinx import mesh as dmesh
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


def test_create_parallelogram_mesh():
    domain = DomainConfig(
        type="parallelogram",
        params={
            "origin": [0.0, 0.0],
            "axis_x": [1.0, 0.0],
            "axis_y": [0.35, 1.0],
            "mesh_resolution": [8, 8],
        },
    )
    domain_geom = create_domain(domain)
    assert domain_geom.mesh.topology.dim == 2
    assert domain_geom.mesh.geometry.dim == 2
    assert set(domain_geom.boundary_names) == {"x-", "x+", "y-", "y+"}
    assert domain_geom.has_periodic_maps is True
    x_periodic = domain_geom.periodic_map("x-", "x+")
    y_periodic = domain_geom.periodic_map("y-", "y+")
    np.testing.assert_allclose(x_periodic.offset, np.array([-1.0, 0.0]))
    np.testing.assert_allclose(y_periodic.offset, np.array([-0.35, -1.0]))


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_create_annulus_mesh():
    domain = DomainConfig(
        type="annulus",
        params={
            "center": [0.2, -0.1],
            "inner_radius": 0.3,
            "outer_radius": 1.0,
            "mesh_size": 0.2,
        },
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
        params={
            "center": [0.2, -0.1],
            "inner_radius": 0.3,
            "outer_radius": 1.0,
            "mesh_size": 0.2,
        },
    )
    domain_geom = create_domain(domain)
    assert set(domain_geom.boundary_names.keys()) == {"inner", "outer"}
    for name, tag in domain_geom.boundary_names.items():
        facets = domain_geom.facet_tags.find(tag)
        assert len(facets) > 0, f"No facets tagged for boundary '{name}'"


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
@pytest.mark.parametrize(
    ("domain_type", "params", "expected_boundaries"),
    [
        (
            "disk",
            {
                "center": [0.5, 0.5],
                "radius": 0.45,
                "mesh_size": 0.2,
            },
            {"outer"},
        ),
        (
            "dumbbell",
            {
                "left_center": [0.35, 0.5],
                "right_center": [0.85, 0.5],
                "lobe_radius": 0.22,
                "neck_width": 0.12,
                "mesh_size": 0.2,
            },
            {"outer"},
        ),
        (
            "l_shape",
            {
                "outer_width": 1.1,
                "outer_height": 1.0,
                "cutout_width": 0.4,
                "cutout_height": 0.35,
                "mesh_size": 0.12,
            },
            {"outer", "notch"},
        ),
        (
            "channel_obstacle",
            {
                "length": 1.4,
                "height": 1.0,
                "obstacle_center": [0.55, 0.5],
                "obstacle_radius": 0.12,
                "mesh_size": 0.2,
            },
            {"inlet", "outlet", "walls", "obstacle"},
        ),
        (
            "airfoil_channel",
            {
                "length": 2.6,
                "height": 1.1,
                "airfoil_center": [1.2, 0.55],
                "chord_length": 0.58,
                "thickness_ratio": 0.13,
                "attack_angle_degrees": 7.0,
                "mesh_size": 0.08,
            },
            {"inlet", "outlet", "walls", "airfoil"},
        ),
        (
            "y_bifurcation",
            {
                "inlet_length": 1.0,
                "branch_length": 0.9,
                "branch_angle_degrees": 38.0,
                "channel_width": 0.24,
                "mesh_size": 0.1,
            },
            {"inlet", "outlet_upper", "outlet_lower", "walls"},
        ),
        (
            "venturi_channel",
            {
                "length": 2.0,
                "height": 1.0,
                "throat_height": 0.36,
                "constriction_center_x": 1.0,
                "constriction_radius": 0.55,
                "mesh_size": 0.08,
            },
            {"inlet", "outlet", "walls"},
        ),
        (
            "serpentine_channel",
            {
                "channel_length": 1.0,
                "lane_spacing": 0.42,
                "n_bends": 3,
                "channel_width": 0.18,
                "mesh_size": 0.08,
            },
            {"inlet", "outlet", "walls"},
        ),
        (
            "side_cavity_channel",
            {
                "length": 2.4,
                "height": 0.8,
                "cavity_width": 0.5,
                "cavity_depth": 0.36,
                "cavity_center_x": 1.2,
                "mesh_size": 0.08,
            },
            {"inlet", "outlet", "walls"},
        ),
    ],
)
def test_gmsh_domain_boundary_tags(domain_type, params, expected_boundaries):
    domain_geom = create_domain(DomainConfig(type=domain_type, params=params))
    assert domain_geom.mesh.topology.dim == 2
    assert domain_geom.mesh.geometry.dim == 2
    assert set(domain_geom.boundary_names.keys()) == expected_boundaries
    for name, tag in domain_geom.boundary_names.items():
        facets = domain_geom.facet_tags.find(tag)
        assert len(facets) > 0, f"No facets tagged for boundary '{name}'"


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_y_bifurcation_outlet_positions():
    domain_geom = create_domain(
        DomainConfig(
            type="y_bifurcation",
            params={
                "inlet_length": 1.0,
                "branch_length": 0.9,
                "branch_angle_degrees": 38.0,
                "channel_width": 0.24,
                "mesh_size": 0.1,
            },
        )
    )
    fdim = domain_geom.mesh.topology.dim - 1
    upper_facets = domain_geom.facet_tags.find(
        domain_geom.boundary_names["outlet_upper"]
    )
    lower_facets = domain_geom.facet_tags.find(
        domain_geom.boundary_names["outlet_lower"]
    )
    upper_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, upper_facets)
    lower_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, lower_facets)
    assert np.mean(upper_midpoints[:, 1]) > 0.0
    assert np.mean(lower_midpoints[:, 1]) < 0.0


def test_venturi_channel_has_narrower_throat_than_inlet():
    domain_geom = create_domain(
        DomainConfig(
            type="venturi_channel",
            params={
                "length": 2.0,
                "height": 1.0,
                "throat_height": 0.36,
                "constriction_center_x": 1.0,
                "constriction_radius": 0.55,
                "mesh_size": 0.08,
            },
        )
    )
    fdim = domain_geom.mesh.topology.dim - 1
    inlet_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["inlet"])
    wall_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["walls"])
    inlet_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, inlet_facets)
    wall_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, wall_facets)
    throat_wall_midpoints = wall_midpoints[np.abs(wall_midpoints[:, 0] - 1.0) < 0.12]
    assert len(throat_wall_midpoints) > 0
    inlet_height = np.max(inlet_midpoints[:, 1]) - np.min(inlet_midpoints[:, 1])
    throat_height = np.max(throat_wall_midpoints[:, 1]) - np.min(
        throat_wall_midpoints[:, 1]
    )
    assert throat_height < inlet_height


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_airfoil_channel_airfoil_boundary_is_interior():
    domain_geom = create_domain(
        DomainConfig(
            type="airfoil_channel",
            params={
                "length": 2.6,
                "height": 1.1,
                "airfoil_center": [1.2, 0.55],
                "chord_length": 0.58,
                "thickness_ratio": 0.13,
                "attack_angle_degrees": 7.0,
                "mesh_size": 0.08,
            },
        )
    )
    fdim = domain_geom.mesh.topology.dim - 1
    airfoil_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["airfoil"])
    wall_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["walls"])
    airfoil_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, airfoil_facets)
    wall_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, wall_facets)
    assert np.min(airfoil_midpoints[:, 0]) > 0.0
    assert np.max(airfoil_midpoints[:, 0]) < 2.6
    assert np.min(airfoil_midpoints[:, 1]) > 0.0
    assert np.max(airfoil_midpoints[:, 1]) < 1.1
    assert np.max(wall_midpoints[:, 1]) - np.min(wall_midpoints[:, 1]) > 1.0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_side_cavity_channel_extends_above_main_channel():
    height = 0.8
    cavity_depth = 0.36
    domain_geom = create_domain(
        DomainConfig(
            type="side_cavity_channel",
            params={
                "length": 2.4,
                "height": height,
                "cavity_width": 0.5,
                "cavity_depth": cavity_depth,
                "cavity_center_x": 1.2,
                "mesh_size": 0.08,
            },
        )
    )
    fdim = domain_geom.mesh.topology.dim - 1
    wall_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["walls"])
    wall_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, wall_facets)
    cavity_roof_midpoints = wall_midpoints[
        wall_midpoints[:, 1] > height + 0.8 * cavity_depth
    ]
    assert len(cavity_roof_midpoints) > 0
    assert np.max(wall_midpoints[:, 1]) > height


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_serpentine_channel_spans_multiple_lanes():
    domain_geom = create_domain(
        DomainConfig(
            type="serpentine_channel",
            params={
                "channel_length": 1.0,
                "lane_spacing": 0.42,
                "n_bends": 3,
                "channel_width": 0.18,
                "mesh_size": 0.08,
            },
        )
    )
    fdim = domain_geom.mesh.topology.dim - 1
    inlet_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["inlet"])
    outlet_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["outlet"])
    wall_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["walls"])
    inlet_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, inlet_facets)
    outlet_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, outlet_facets)
    wall_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, wall_facets)
    assert np.mean(outlet_midpoints[:, 1]) > np.mean(inlet_midpoints[:, 1])
    assert np.max(wall_midpoints[:, 1]) - np.min(wall_midpoints[:, 1]) > 1.0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
def test_l_shape_notch_boundary_tracks_reentrant_corner():
    outer_width = 1.2
    outer_height = 1.0
    cutout_width = 0.45
    cutout_height = 0.35
    domain_geom = create_domain(
        DomainConfig(
            type="l_shape",
            params={
                "outer_width": outer_width,
                "outer_height": outer_height,
                "cutout_width": cutout_width,
                "cutout_height": cutout_height,
                "mesh_size": 0.1,
            },
        )
    )
    reentrant_x = outer_width - cutout_width
    reentrant_y = outer_height - cutout_height
    fdim = domain_geom.mesh.topology.dim - 1
    notch_facets = domain_geom.facet_tags.find(domain_geom.boundary_names["notch"])
    notch_midpoints = dmesh.compute_midpoints(domain_geom.mesh, fdim, notch_facets)

    assert np.any(np.isclose(notch_midpoints[:, 0], reentrant_x, atol=0.08))
    assert np.any(np.isclose(notch_midpoints[:, 1], reentrant_y, atol=0.08))
    assert np.all(
        np.isclose(notch_midpoints[:, 0], reentrant_x, atol=0.08)
        | np.isclose(notch_midpoints[:, 1], reentrant_y, atol=0.08)
    )


@pytest.mark.parametrize(
    ("domain_type", "params", "match"),
    [
        (
            "annulus",
            {
                "center": [0.0, 0.0],
                "inner_radius": 0.0,
                "outer_radius": 1.0,
                "mesh_size": 0.2,
            },
            "inner_radius",
        ),
        (
            "annulus",
            {
                "center": [0.0, 0.0],
                "inner_radius": 0.3,
                "outer_radius": -1.0,
                "mesh_size": 0.2,
            },
            "outer_radius",
        ),
        (
            "annulus",
            {
                "center": [0.0, 0.0],
                "inner_radius": 1.0,
                "outer_radius": 1.0,
                "mesh_size": 0.2,
            },
            "inner_radius' < 'outer_radius",
        ),
        (
            "annulus",
            {
                "center": [0.0, 0.0],
                "inner_radius": 0.3,
                "outer_radius": 1.0,
                "mesh_size": 0.0,
            },
            "mesh_size",
        ),
        (
            "disk",
            {"center": [0.5, 0.5], "radius": 0.0, "mesh_size": 0.2},
            "radius",
        ),
        (
            "dumbbell",
            {
                "left_center": [0.5, 0.5],
                "right_center": [0.4, 0.5],
                "lobe_radius": 0.2,
                "neck_width": 0.1,
                "mesh_size": 0.2,
            },
            "right_center",
        ),
        (
            "dumbbell",
            {
                "left_center": [0.3, 0.4],
                "right_center": [0.8, 0.5],
                "lobe_radius": 0.2,
                "neck_width": 0.1,
                "mesh_size": 0.2,
            },
            "same y-coordinate",
        ),
        (
            "l_shape",
            {
                "outer_width": 1.0,
                "outer_height": 1.0,
                "cutout_width": 1.0,
                "cutout_height": 0.3,
                "mesh_size": 0.2,
            },
            "cutout_width",
        ),
        (
            "l_shape",
            {
                "outer_width": 1.0,
                "outer_height": 1.0,
                "cutout_width": 0.3,
                "cutout_height": 1.0,
                "mesh_size": 0.2,
            },
            "cutout_height",
        ),
        (
            "parallelogram",
            {
                "origin": [0.0, 0.0],
                "axis_x": [1.0, 0.0],
                "axis_y": [2.0, 0.0],
                "mesh_resolution": [8, 8],
            },
            "linearly independent",
        ),
        (
            "channel_obstacle",
            {
                "length": 1.0,
                "height": 1.0,
                "obstacle_center": [0.08, 0.5],
                "obstacle_radius": 0.12,
                "mesh_size": 0.2,
            },
            "strictly inside the channel in x",
        ),
        (
            "airfoil_channel",
            {
                "length": 2.6,
                "height": 1.1,
                "airfoil_center": [0.25, 0.55],
                "chord_length": 0.58,
                "thickness_ratio": 0.13,
                "attack_angle_degrees": 0.0,
                "mesh_size": 0.08,
            },
            "strictly inside the channel in x",
        ),
        (
            "airfoil_channel",
            {
                "length": 2.6,
                "height": 1.1,
                "airfoil_center": [1.2, 0.02],
                "chord_length": 0.58,
                "thickness_ratio": 0.13,
                "attack_angle_degrees": 0.0,
                "mesh_size": 0.08,
            },
            "strictly inside the channel in y",
        ),
        (
            "airfoil_channel",
            {
                "length": 2.6,
                "height": 1.1,
                "airfoil_center": [1.2, 0.55],
                "chord_length": 0.58,
                "thickness_ratio": 0.28,
                "attack_angle_degrees": 0.0,
                "mesh_size": 0.08,
            },
            "thickness_ratio",
        ),
        (
            "y_bifurcation",
            {
                "inlet_length": 1.0,
                "branch_length": 0.3,
                "branch_angle_degrees": 25.0,
                "channel_width": 0.24,
                "mesh_size": 0.1,
            },
            "separate cleanly",
        ),
        (
            "y_bifurcation",
            {
                "inlet_length": 1.0,
                "branch_length": 0.9,
                "branch_angle_degrees": 90.0,
                "channel_width": 0.24,
                "mesh_size": 0.1,
            },
            "branch_angle_degrees",
        ),
        (
            "venturi_channel",
            {
                "length": 2.0,
                "height": 1.0,
                "throat_height": 1.0,
                "constriction_center_x": 1.0,
                "constriction_radius": 0.55,
                "mesh_size": 0.08,
            },
            "throat_height",
        ),
        (
            "venturi_channel",
            {
                "length": 2.0,
                "height": 1.0,
                "throat_height": 0.12,
                "constriction_center_x": 1.0,
                "constriction_radius": 0.4,
                "mesh_size": 0.08,
            },
            "constriction_radius",
        ),
        (
            "venturi_channel",
            {
                "length": 2.0,
                "height": 1.0,
                "throat_height": 0.36,
                "constriction_center_x": 0.35,
                "constriction_radius": 0.55,
                "mesh_size": 0.08,
            },
            "strictly inside the channel in x",
        ),
        (
            "serpentine_channel",
            {
                "channel_length": 0.18,
                "lane_spacing": 0.42,
                "n_bends": 3,
                "channel_width": 0.18,
                "mesh_size": 0.08,
            },
            "channel_length",
        ),
        (
            "serpentine_channel",
            {
                "channel_length": 1.0,
                "lane_spacing": 0.18,
                "n_bends": 3,
                "channel_width": 0.18,
                "mesh_size": 0.08,
            },
            "lane_spacing",
        ),
        (
            "serpentine_channel",
            {
                "channel_length": 1.0,
                "lane_spacing": 0.42,
                "n_bends": 1,
                "channel_width": 0.18,
                "mesh_size": 0.08,
            },
            "n_bends",
        ),
        (
            "side_cavity_channel",
            {
                "length": 2.0,
                "height": 0.8,
                "cavity_width": 2.0,
                "cavity_depth": 0.36,
                "cavity_center_x": 1.0,
                "mesh_size": 0.08,
            },
            "cavity_width",
        ),
    ],
)
def test_domain_parameter_validation(domain_type, params, match):
    with pytest.raises(ValueError, match=match):
        create_domain(DomainConfig(type=domain_type, params=params))


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
