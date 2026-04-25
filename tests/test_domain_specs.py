"""Tests for first-class domain specifications."""

from plm_data.core.runtime_config import DomainConfig
from plm_data.domains import create_domain
from plm_data.domains import (
    get_gmsh_domain_dimension,
    get_domain_spec,
    is_gmsh_domain,
    is_gmsh_planar_domain,
    list_domain_specs,
    list_domains,
)


def test_all_current_domains_have_specs():
    specs = list_domain_specs()
    expected = {
        "airfoil_channel",
        "annulus",
        "channel_obstacle",
        "disk",
        "dumbbell",
        "l_shape",
        "multi_hole_plate",
        "parallelogram",
        "porous_channel",
        "rectangle",
        "serpentine_channel",
        "side_cavity_channel",
        "venturi_channel",
        "y_bifurcation",
    }

    assert set(specs) == expected
    for name, spec in specs.items():
        assert spec.name == name
        assert spec.dimension == 2
        assert spec.boundary_names
        assert spec.boundary_roles["all"]
        assert set(spec.boundary_roles["all"]).issubset(set(spec.boundary_names))
        assert set(spec.parameters) == {
            param.name for param in spec.parameters.values()
        }


def test_rectangle_spec_exposes_boundary_roles_and_periodic_pairs():
    spec = get_domain_spec("rectangle")

    assert spec.dimension == 2
    assert spec.boundary_names == ("x-", "x+", "y-", "y+")
    assert spec.boundary_roles["x_pair"] == ("x-", "x+")
    assert spec.boundary_roles["walls"] == ("x-", "x+", "y-", "y+")
    assert spec.periodic_pairs == (("x-", "x+"), ("y-", "y+"))
    assert "scalar_all_neumann" in spec.supported_boundary_scenarios
    assert "heat_gaussian_bump" in spec.supported_initial_condition_scenarios
    assert "center" in spec.coordinate_regions
    assert "left_half" in spec.coordinate_regions


def test_domain_registry_keeps_moved_rectangle_factory_available():
    assert "rectangle" in list_domains()

    domain_geom = create_domain(
        DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [4, 4]},
        )
    )

    assert set(domain_geom.boundary_names) == {"x-", "x+", "y-", "y+"}
    assert domain_geom.has_periodic_maps is True


def test_gmsh_planar_domain_lookup_loads_domain_modules():
    assert is_gmsh_planar_domain("disk") is True


def test_all_registered_domains_are_gmsh_backed():
    for name in list_domains():
        assert is_gmsh_domain(name) is True
        assert get_gmsh_domain_dimension(name) == get_domain_spec(name).dimension
