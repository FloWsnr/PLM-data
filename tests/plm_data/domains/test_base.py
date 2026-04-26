"""Tests for first-class domain specifications."""

from types import SimpleNamespace

import pytest

from plm_data.core.runtime_config import DomainConfig
import plm_data.domains.base as domain_base
from plm_data.domains import create_domain
from plm_data.domains import (
    get_gmsh_domain_dimension,
    get_domain_spec,
    is_gmsh_domain,
    is_gmsh_planar_domain,
    list_domain_specs,
    list_domains,
)
from plm_data.domains.base import DomainParameterSpec, DomainSpec, register_domain_spec


def test_register_domain_spec_adds_spec(monkeypatch):
    registry: dict[str, DomainSpec] = {}
    monkeypatch.setattr(domain_base, "_DOMAIN_SPEC_REGISTRY", registry)
    spec = DomainSpec(
        name="unit_test",
        dimension=2,
        parameters={},
        boundary_names=("outer",),
        boundary_roles={"all": ("outer",)},
    )

    registered = register_domain_spec(spec)

    assert registered is spec
    assert registry == {"unit_test": spec}


def test_domain_parameter_spec_rejects_invalid_sampling_bounds():
    with pytest.raises(ValueError, match="both sampling_min and sampling_max"):
        DomainParameterSpec(
            name="bad",
            kind="float",
            sampling_min=0.0,
        )

    with pytest.raises(ValueError, match="sampling_min greater than sampling_max"):
        DomainParameterSpec(
            name="bad",
            kind="float",
            sampling_min=2.0,
            sampling_max=1.0,
        )

    with pytest.raises(ValueError, match="sampling_max must be <= hard_max"):
        DomainParameterSpec(
            name="bad",
            kind="float",
            hard_max=1.0,
            sampling_min=0.2,
            sampling_max=1.2,
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


def test_register_gmsh_planar_domain_registers_2d_builder(monkeypatch):
    registry: dict[str, object] = {}
    monkeypatch.setattr(domain_base, "_GMSH_DOMAIN_REGISTRY", registry)
    calls: list[tuple[object, object]] = []

    @domain_base.register_gmsh_planar_domain("unit_test_planar")
    def _builder(model, domain):
        calls.append((model, domain))

    domain = SimpleNamespace(type="unit_test_planar")
    model = object()

    assert domain_base.is_gmsh_planar_domain("unit_test_planar") is True
    domain_base.build_gmsh_planar_domain_model(domain, model)

    assert calls == [(model, domain)]


def test_all_registered_domains_are_gmsh_backed():
    for name in list_domains():
        assert is_gmsh_domain(name) is True
        assert get_gmsh_domain_dimension(name) == get_domain_spec(name).dimension
