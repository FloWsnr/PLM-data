"""Tests for plm_data.core.config."""

import pytest
import yaml

from plm_data.core.config import load_config


def test_load_config():
    cfg = load_config("configs/basic/heat/2d_default.yaml")
    assert cfg.preset == "heat"
    assert cfg.parameters == {}
    assert cfg.coefficient("kappa").type == "constant"
    assert cfg.coefficient("kappa").params["value"] == 0.01
    assert cfg.domain.type == "rectangle"
    assert cfg.output.resolution == [64, 64]
    assert cfg.time.dt == 0.01
    assert cfg.time.t_end == 1.0
    assert cfg.has_periodic_boundary_conditions is False
    assert cfg.input("u").initial_condition.type == "gaussian_bump"
    assert cfg.input("u").source.type == "none"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "neumann"


def test_load_config_periodic_field():
    cfg = load_config("configs/physics/cahn_hilliard/2d_default.yaml")
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.boundary_field("c").side_conditions("x-")[0].pair_with == "x+"
    assert cfg.boundary_field("c").side_conditions("y+")[0].type == "periodic"


def test_load_config_missing_field(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump({"parameters": {"k": 1.0}}))
    with pytest.raises(ValueError, match="preset"):
        load_config(bad_yaml)


def test_load_config_boundary_field_sections():
    cfg = load_config("configs/basic/poisson/2d_default.yaml")
    u_boundary = cfg.boundary_field("u")
    assert set(u_boundary.sides) == {"x-", "x+", "y-", "y+"}
    assert cfg.input("u").source.type == "sine_product"
    assert cfg.output_mode("u") == "scalar"


def test_load_config_vector_input():
    cfg = load_config("configs/fluids/navier_stokes/2d_cavity_re400.yaml")
    velocity = cfg.input("velocity")
    velocity_bcs = cfg.boundary_field("velocity")
    assert cfg.output_mode("velocity") == "components"
    assert cfg.output_mode("pressure") == "scalar"
    assert velocity.source.type == "none"
    assert velocity.initial_condition.type == "custom"
    assert (
        velocity_bcs.side_conditions("y+")[0].value.components["x"].params["value"]
        == 1.0
    )


def test_load_config_thermal_convection_2d():
    cfg = load_config("configs/fluids/thermal_convection/2d_default.yaml")
    assert cfg.preset == "thermal_convection"
    assert cfg.domain.dimension == 2
    assert cfg.output_mode("velocity") == "components"
    assert cfg.output_mode("pressure") == "scalar"
    assert cfg.output_mode("temperature") == "scalar"
    assert cfg.input("velocity").initial_condition.type == "zero"
    assert cfg.input("temperature").initial_condition.type == "random_perturbation"
    assert cfg.boundary_field("velocity").side_conditions("x-")[0].type == "periodic"
    assert (
        cfg.boundary_field("temperature").side_conditions("y-")[0].value.params["value"]
        == 1.0
    )


def test_load_config_thermal_convection_3d():
    cfg = load_config("configs/fluids/thermal_convection/3d_default.yaml")
    assert cfg.preset == "thermal_convection"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.boundary_field("velocity").side_conditions("y+")[0].pair_with == "y-"
    assert (
        cfg.boundary_field("temperature").side_conditions("z+")[0].value.params["value"]
        == 0.0
    )


def test_load_config_maxwell_pulse():
    cfg = load_config("configs/physics/maxwell_pulse/2d_default.yaml")
    electric_field = cfg.input("electric_field")
    boundary_field = cfg.boundary_field("electric_field")
    assert cfg.output_mode("electric_field") == "components"
    assert electric_field.initial_condition.type == "zero"
    assert boundary_field.side_conditions("x-")[0].type == "absorbing"
    assert boundary_field.side_conditions("y-")[0].type == "dirichlet"


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
        "inputs": {
            "u": {
                "source": {"type": "none", "params": {}},
            },
        },
        "boundary_conditions": {
            "u": {
                "x-": [{"operator": "dirichlet", "value": 0.0}],
                "x+": [{"operator": "dirichlet", "value": 0.0}],
                "y-": [{"operator": "dirichlet", "value": 0.0}],
                "y+": [{"operator": "dirichlet", "value": 0.0}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 1,
            "formats": ["numpy"],
            "fields": {"u": "scalar"},
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
        "inputs": {
            "velocity": {
                "source": {"type": "none", "params": {}},
            },
        },
        "boundary_conditions": {
            "velocity": {
                "x-": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "x+": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "y-": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "y+": [{"operator": "dirichlet", "value": [1.0, 0.0]}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 1,
            "formats": ["numpy"],
            "fields": {"velocity": "components", "pressure": "scalar"},
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
        "inputs": {
            "electric_field": {
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
            }
        },
        "boundary_conditions": {
            "electric_field": {
                "x-": [
                    {"operator": "absorbing", "value": {"type": "zero", "params": {}}}
                ],
                "x+": [
                    {"operator": "absorbing", "value": {"type": "zero", "params": {}}}
                ],
                "y-": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "y+": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 2,
            "formats": ["numpy"],
            "fields": {"electric_field": "components"},
        },
        "solver": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        "time": {"dt": 0.05, "t_end": 0.1},
    }


def test_robin_bc_missing_alpha(tmp_path):
    data = _base_yaml_dict()
    data["boundary_conditions"]["u"]["x-"] = [{"operator": "robin", "value": 0.0}]
    p = tmp_path / "robin_no_alpha.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="alpha"):
        load_config(p)


def test_missing_inputs_section(tmp_path):
    data = _base_yaml_dict()
    del data["inputs"]
    p = tmp_path / "no_inputs.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="inputs"):
        load_config(p)


def test_missing_coefficients_section(tmp_path):
    data = {
        "preset": "heat",
        "parameters": {},
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "time": {"dt": 0.01, "t_end": 0.01},
        "inputs": {
            "u": {
                "source": {"type": "none", "params": {}},
                "initial_condition": {"type": "constant", "params": {"value": 0.0}},
            }
        },
        "boundary_conditions": {
            "u": {
                "x-": [{"operator": "neumann", "value": 0.0}],
                "x+": [{"operator": "neumann", "value": 0.0}],
                "y-": [{"operator": "neumann", "value": 0.0}],
                "y+": [{"operator": "neumann", "value": 0.0}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 2,
            "formats": ["numpy"],
            "fields": {"u": "scalar"},
        },
        "solver": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        "seed": 42,
    }
    p = tmp_path / "no_coefficients.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="coefficients"):
        load_config(p)


def test_missing_boundary_conditions_section(tmp_path):
    data = _base_yaml_dict()
    del data["boundary_conditions"]
    p = tmp_path / "no_boundary_conditions.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="boundary_conditions"):
        load_config(p)


def test_missing_output_resolution(tmp_path):
    data = _base_yaml_dict()
    del data["output"]["resolution"]
    p = tmp_path / "no_resolution.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="resolution"):
        load_config(p)


def test_output_path_is_rejected(tmp_path):
    data = _base_yaml_dict()
    data["output"]["path"] = "./output"
    p = tmp_path / "output_path.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="unsupported keys"):
        load_config(p)


def test_periodic_pair_must_be_reciprocal(tmp_path):
    data = _base_yaml_dict()
    data["boundary_conditions"]["u"] = {
        "x-": [{"operator": "periodic", "pair_with": "x+"}],
        "x+": [{"operator": "periodic", "pair_with": "y+"}],
        "y-": [{"operator": "dirichlet", "value": 0.0}],
        "y+": [{"operator": "dirichlet", "value": 0.0}],
    }
    p = tmp_path / "bad_periodic_pair.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="reciprocal"):
        load_config(p)


def test_parse_domain_periodic_map(tmp_path):
    data = _base_yaml_dict()
    data["domain"]["periodic_maps"] = {
        "streamwise": {
            "slave": "x+",
            "master": "x-",
            "transform": {
                "type": "affine",
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
                "offset": [-1.0, 0.0],
            },
        }
    }
    p = tmp_path / "periodic_map.yaml"
    p.write_text(yaml.dump(data))

    cfg = load_config(p)
    assert "streamwise" in cfg.domain.periodic_maps
    assert cfg.domain.periodic_maps["streamwise"].slave == "x+"


def test_load_config_vector_neumann_bc(tmp_path):
    data = _base_stokes_yaml_dict()
    data["boundary_conditions"]["velocity"]["x-"] = [
        {
            "operator": "neumann",
            "value": [1.0, 0.0],
        }
    ]
    p = tmp_path / "stokes_vector_neumann.yaml"
    p.write_text(yaml.dump(data))

    cfg = load_config(p)
    bc = cfg.boundary_field("velocity").side_conditions("x-")[0]
    assert bc.type == "neumann"
    assert bc.value.components["x"].params["value"] == 1.0
    assert bc.value.components["y"].params["value"] == 0.0


def test_vector_robin_bc_rejected_at_parse_time(tmp_path):
    data = _base_stokes_yaml_dict()
    data["boundary_conditions"]["velocity"]["x-"] = [
        {
            "operator": "robin",
            "value": [1.0, 0.0],
            "operator_parameters": {"alpha": 1.0},
        }
    ]
    p = tmp_path / "stokes_vector_robin.yaml"
    p.write_text(yaml.dump(data))

    with pytest.raises(ValueError, match="unsupported operator"):
        load_config(p)


def test_load_config_vector_absorbing_bc(tmp_path):
    data = _base_maxwell_pulse_yaml_dict()
    p = tmp_path / "maxwell_pulse_absorbing.yaml"
    p.write_text(yaml.dump(data))

    cfg = load_config(p)
    bc = cfg.boundary_field("electric_field").side_conditions("x-")[0]
    assert bc.type == "absorbing"
    assert bc.value.type == "zero"
