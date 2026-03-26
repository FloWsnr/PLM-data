"""Tests for plm_data.core.config."""

import pytest
import yaml

from plm_data.core.config import load_config


def test_load_config():
    cfg = load_config("configs/basic/heat/2d_default.yaml")
    assert cfg.preset == "heat"
    assert "kappa" in cfg.parameters
    assert cfg.domain.type == "rectangle"
    assert cfg.output.resolution == [64, 64]
    assert cfg.time.dt == 0.01
    assert cfg.time.t_end == 1.0
    assert cfg.field("u").initial_condition.type == "gaussian_bump"
    assert cfg.field("u").source.type == "none"


def test_load_config_missing_field(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump({"parameters": {"k": 1.0}}))
    with pytest.raises(ValueError, match="preset"):
        load_config(bad_yaml)


def test_load_config_field_sections():
    cfg = load_config("configs/basic/poisson/2d_default.yaml")
    u_field = cfg.field("u")
    assert "x-" in u_field.boundary_conditions
    assert "x+" in u_field.boundary_conditions
    assert "y-" in u_field.boundary_conditions
    assert "y+" in u_field.boundary_conditions
    assert u_field.source.type == "sine_product"
    assert u_field.output.mode == "scalar"


def test_load_config_vector_field():
    cfg = load_config("configs/fluids/navier_stokes/2d_default.yaml")
    velocity = cfg.field("velocity")
    assert velocity.output.mode == "components"
    assert velocity.source.type == "none"
    assert velocity.initial_condition.type == "custom"
    assert (
        velocity.boundary_conditions["y+"].value.components["x"].params["value"] == 1.0
    )


def test_load_config_maxwell_pulse():
    cfg = load_config("configs/physics/maxwell_pulse/2d_default.yaml")
    electric_field = cfg.field("electric_field")
    assert electric_field.output.mode == "components"
    assert electric_field.initial_condition.type == "zero"
    assert electric_field.boundary_conditions["x-"].type == "absorbing"
    assert electric_field.boundary_conditions["y-"].type == "dirichlet"


def _base_yaml_dict():
    """Return a minimal valid config dict with all required top-level fields."""
    return {
        "preset": "poisson",
        "parameters": {"kappa": 1.0, "f_amplitude": 1.0},
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "fields": {
            "u": {
                "boundary_conditions": {
                    "x-": {"type": "dirichlet", "value": 0.0},
                },
                "source": {"type": "none", "params": {}},
                "output": "scalar",
            },
        },
        "output": {
            "path": "./output",
            "resolution": [8, 8],
            "num_frames": 1,
            "formats": ["numpy"],
        },
        "solver": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    }


def _base_stokes_yaml_dict():
    return {
        "preset": "stokes",
        "parameters": {"nu": 1.0},
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "fields": {
            "velocity": {
                "boundary_conditions": {
                    "x-": {"type": "dirichlet", "value": [0.0, 0.0]},
                    "x+": {"type": "dirichlet", "value": [0.0, 0.0]},
                    "y-": {"type": "dirichlet", "value": [0.0, 0.0]},
                    "y+": {"type": "dirichlet", "value": [1.0, 0.0]},
                },
                "source": {"type": "none", "params": {}},
                "output": "components",
            },
            "pressure": {"output": "scalar"},
        },
        "output": {
            "path": "./output",
            "resolution": [8, 8],
            "num_frames": 1,
            "formats": ["numpy"],
        },
        "solver": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    }


def _base_maxwell_pulse_yaml_dict():
    return {
        "preset": "maxwell_pulse",
        "parameters": {
            "epsilon_r": 1.0,
            "mu_r": 1.0,
            "sigma": 0.05,
            "pulse_amplitude": 1.0,
            "pulse_frequency": 4.0,
            "pulse_width": 0.1,
            "pulse_delay": 0.2,
        },
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "fields": {
            "electric_field": {
                "boundary_conditions": {
                    "x-": {"type": "absorbing", "value": {"type": "zero", "params": {}}},
                    "x+": {"type": "absorbing", "value": {"type": "zero", "params": {}}},
                    "y-": {"type": "dirichlet", "value": [0.0, 0.0]},
                    "y+": {"type": "dirichlet", "value": [0.0, 0.0]},
                },
                "source": {
                    "components": {
                        "x": {
                            "type": "gaussian_bump",
                            "params": {
                                "amplitude": 1.0,
                                "sigma": 0.1,
                                "center": [0.5, 0.5],
                            },
                        },
                        "y": {"type": "zero", "params": {}},
                    }
                },
                "initial_condition": {"type": "zero", "params": {}},
                "output": "components",
            }
        },
        "output": {
            "path": "./output",
            "resolution": [8, 8],
            "num_frames": 2,
            "formats": ["numpy"],
        },
        "solver": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        "time": {"dt": 0.05, "t_end": 0.1},
    }


def test_robin_bc_missing_alpha(tmp_path):
    data = _base_yaml_dict()
    data["fields"]["u"]["boundary_conditions"]["x-"] = {"type": "robin", "value": 0.0}
    p = tmp_path / "robin_no_alpha.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="alpha"):
        load_config(p)


def test_missing_fields_section(tmp_path):
    data = _base_yaml_dict()
    del data["fields"]
    p = tmp_path / "no_fields.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="fields"):
        load_config(p)


def test_missing_output_resolution(tmp_path):
    data = _base_yaml_dict()
    del data["output"]["resolution"]
    p = tmp_path / "no_resolution.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="resolution"):
        load_config(p)


def test_load_config_vector_neumann_bc(tmp_path):
    data = _base_stokes_yaml_dict()
    data["fields"]["velocity"]["boundary_conditions"]["x-"] = {
        "type": "neumann",
        "value": [1.0, 0.0],
    }
    p = tmp_path / "stokes_vector_neumann.yaml"
    p.write_text(yaml.dump(data))

    cfg = load_config(p)
    bc = cfg.field("velocity").boundary_conditions["x-"]
    assert bc.type == "neumann"
    assert bc.value.components["x"].params["value"] == 1.0
    assert bc.value.components["y"].params["value"] == 0.0


def test_vector_robin_bc_still_rejected(tmp_path):
    data = _base_stokes_yaml_dict()
    data["fields"]["velocity"]["boundary_conditions"]["x-"] = {
        "type": "robin",
        "value": [1.0, 0.0],
        "alpha": 1.0,
    }
    p = tmp_path / "stokes_vector_robin.yaml"
    p.write_text(yaml.dump(data))

    with pytest.raises(ValueError, match="only supported for scalar fields"):
        load_config(p)


def test_load_config_vector_absorbing_bc(tmp_path):
    data = _base_maxwell_pulse_yaml_dict()
    p = tmp_path / "maxwell_pulse_absorbing.yaml"
    p.write_text(yaml.dump(data))

    cfg = load_config(p)
    bc = cfg.field("electric_field").boundary_conditions["x-"]
    assert bc.type == "absorbing"
    assert bc.value.type == "zero"
