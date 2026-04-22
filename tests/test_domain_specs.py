"""Tests for first-class domain specifications."""

from plm_data.core.config import DomainConfig
from plm_data.core.mesh import create_domain
from plm_data.domains import (
    get_domain_spec,
    is_gmsh_planar_domain,
    list_domain_specs,
    list_domains,
)


def test_all_current_domains_have_specs():
    specs = list_domain_specs()
    expected = {
        "airfoil_channel",
        "annulus",
        "box",
        "channel_obstacle",
        "disk",
        "dumbbell",
        "interval",
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
        assert spec.dimension in {1, 2, 3}
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
    assert "periodic_axis" in spec.allowed_boundary_families
    assert "gaussian_blobs" in spec.allowed_initial_condition_families
    assert "radial_cosine" in spec.allowed_initial_condition_families


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


def test_gmsh_planar_domain_lookup_loads_legacy_registrations():
    assert is_gmsh_planar_domain("disk") is True
