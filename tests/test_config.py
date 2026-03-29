"""Tests for plm_data.core.config."""

import pytest
import yaml

from plm_data.core.config import load_config
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    STATIONARY_SCALAR_SPD,
    STEADY_SADDLE_POINT,
    TRANSIENT_MIXED_DIRECT,
)


def _solver_block(
    strategy: str,
    *,
    serial: dict[str, str] | None = None,
    mpi: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "serial": {"ksp_type": "preonly", "pc_type": "lu"}
        if serial is None
        else serial,
        "mpi": {"ksp_type": "preonly", "pc_type": "lu"} if mpi is None else mpi,
    }


def _write_yaml(tmp_path, name: str, data: dict[str, object]):
    path = tmp_path / name
    path.write_text(yaml.dump(data))
    return path


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
    assert cfg.solver.strategy == CONSTANT_LHS_SCALAR_SPD
    assert cfg.solver.profile_name == "serial"
    assert cfg.solver.serial["pc_type"] == "lu"
    assert cfg.solver.mpi["pc_type"] == "hypre"


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
    assert cfg.solver.strategy == STATIONARY_SCALAR_SPD


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
    assert cfg.solver.profile_name == "serial"


def test_load_config_thermal_convection_2d():
    cfg = load_config("configs/fluids/thermal_convection/2d_default.yaml")
    assert cfg.preset == "thermal_convection"
    assert cfg.domain.dimension == 2
    assert cfg.output_mode("velocity") == "components"
    assert cfg.output_mode("pressure") == "scalar"
    assert cfg.output_mode("temperature") == "scalar"
    assert cfg.input("velocity").initial_condition.type == "zero"
    assert cfg.input("temperature").initial_condition.type == "conductive_noise"
    assert cfg.input("temperature").initial_condition.params["amplitude"] == 0.15
    assert cfg.boundary_field("velocity").side_conditions("x-")[0].type == "periodic"
    assert (
        cfg.boundary_field("temperature").side_conditions("y-")[0].value.params["value"]
        == 1.0
    )
    assert cfg.solver.strategy == TRANSIENT_MIXED_DIRECT


def test_load_config_thermal_convection_3d():
    cfg = load_config("configs/fluids/thermal_convection/3d_default.yaml")
    assert cfg.preset == "thermal_convection"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.input("temperature").initial_condition.type == "conductive_noise"
    assert cfg.boundary_field("velocity").side_conditions("y+")[0].pair_with == "y-"
    assert (
        cfg.boundary_field("temperature").side_conditions("z+")[0].value.params["value"]
        == 0.0
    )


def test_load_config_advection_2d():
    cfg = load_config("configs/basic/advection/2d_default.yaml")
    assert cfg.preset == "advection"
    assert cfg.domain.dimension == 2
    assert cfg.output_mode("u") == "scalar"
    assert cfg.input("u").initial_condition.type == "gaussian_bump"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "periodic"
    assert cfg.coefficient("diffusivity").params["value"] == 0.0005
    velocity = cfg.coefficient("velocity")
    assert velocity.components["x"].type == "sine_product"
    assert velocity.components["y"].params["amplitude"] == -1.25
    assert cfg.solver.strategy == CONSTANT_LHS_SCALAR_NONSYMMETRIC
    assert cfg.solver.mpi["ksp_type"] == "gmres"


def test_load_config_advection_3d():
    cfg = load_config("configs/basic/advection/3d_default.yaml")
    assert cfg.preset == "advection"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.coefficient("velocity").components["z"].params["kx"] == 2.0
    assert cfg.boundary_field("u").side_conditions("z+")[0].pair_with == "z-"


def test_load_config_maxwell_pulse():
    cfg = load_config("configs/physics/maxwell_pulse/2d_default.yaml")
    electric_field = cfg.input("electric_field")
    boundary_field = cfg.boundary_field("electric_field")
    assert cfg.output_mode("electric_field") == "components"
    assert electric_field.initial_condition.type == "zero"
    assert boundary_field.side_conditions("x-")[0].type == "absorbing"
    assert boundary_field.side_conditions("y-")[0].type == "dirichlet"
    assert cfg.solver.strategy == CONSTANT_LHS_CURL_DIRECT


def test_load_config_wave():
    cfg = load_config("configs/basic/wave/2d_default.yaml")
    boundary_field = cfg.boundary_field("u")
    assert cfg.preset == "wave"
    assert cfg.parameters["damping"] == 0.03
    assert cfg.coefficient("c_sq").type == "constant"
    assert cfg.coefficient("c_sq").params["value"] == 4.0
    assert cfg.input("u").initial_condition.type == "zero"
    assert cfg.input("v").initial_condition.type == "gaussian_bump"
    assert cfg.input("forcing").source.type == "none"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.output_mode("v") == "scalar"
    assert boundary_field.side_conditions("x-")[0].type == "dirichlet"
    assert boundary_field.side_conditions("y-")[0].type == "neumann"


def test_load_config_plate():
    cfg = load_config("configs/basic/plate/2d_default.yaml")
    boundary_field = cfg.boundary_field("deflection")
    assert cfg.preset == "plate"
    assert cfg.parameters["theta"] == 0.5
    assert cfg.coefficient("rho_h").type == "constant"
    assert cfg.coefficient("damping").params["value"] == 0.0
    assert cfg.coefficient("rigidity").params["value"] == 0.2
    assert cfg.input("deflection").initial_condition.type == "sine_product"
    assert cfg.input("deflection").initial_condition.params["amplitude"] == 0.1
    assert cfg.input("velocity").initial_condition.type == "zero"
    assert cfg.input("load").source.type == "none"
    assert cfg.output_mode("deflection") == "scalar"
    assert cfg.output_mode("velocity") == "scalar"
    assert boundary_field.side_conditions("x-")[0].type == "simply_supported"


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
            **_solver_block(STATIONARY_SCALAR_SPD),
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
            **_solver_block(STEADY_SADDLE_POINT),
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
            **_solver_block(CONSTANT_LHS_CURL_DIRECT),
        },
        "time": {"dt": 0.05, "t_end": 0.1},
    }


def test_robin_bc_missing_alpha(tmp_path):
    data = _base_yaml_dict()
    data["boundary_conditions"]["u"]["x-"] = [{"operator": "robin", "value": 0.0}]
    p = _write_yaml(tmp_path, "robin_no_alpha.yaml", data)
    with pytest.raises(ValueError, match="alpha"):
        load_config(p)


def test_missing_inputs_section(tmp_path):
    data = _base_yaml_dict()
    del data["inputs"]
    p = _write_yaml(tmp_path, "no_inputs.yaml", data)
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
            **_solver_block(CONSTANT_LHS_SCALAR_SPD),
        },
        "seed": 42,
    }
    p = _write_yaml(tmp_path, "no_coefficients.yaml", data)
    with pytest.raises(ValueError, match="coefficients"):
        load_config(p)


def test_missing_boundary_conditions_section(tmp_path):
    data = _base_yaml_dict()
    del data["boundary_conditions"]
    p = _write_yaml(tmp_path, "no_boundary_conditions.yaml", data)
    with pytest.raises(ValueError, match="boundary_conditions"):
        load_config(p)


def test_missing_solver_strategy(tmp_path):
    data = _base_yaml_dict()
    del data["solver"]["strategy"]
    p = _write_yaml(tmp_path, "no_strategy.yaml", data)
    with pytest.raises(ValueError, match="Missing required field 'strategy' in solver"):
        load_config(p)


def test_missing_solver_profile(tmp_path):
    data = _base_yaml_dict()
    del data["solver"]["mpi"]
    p = _write_yaml(tmp_path, "no_solver_mpi.yaml", data)
    with pytest.raises(ValueError, match="Missing required field 'mpi' in solver"):
        load_config(p)


def test_missing_output_resolution(tmp_path):
    data = _base_yaml_dict()
    del data["output"]["resolution"]
    p = _write_yaml(tmp_path, "no_resolution.yaml", data)
    with pytest.raises(ValueError, match="resolution"):
        load_config(p)


def test_output_path_is_rejected(tmp_path):
    data = _base_yaml_dict()
    data["output"]["path"] = "./output"
    p = _write_yaml(tmp_path, "output_path.yaml", data)
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
    p = _write_yaml(tmp_path, "bad_periodic_pair.yaml", data)
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
    p = _write_yaml(tmp_path, "periodic_map.yaml", data)

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
    p = _write_yaml(tmp_path, "stokes_vector_neumann.yaml", data)

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
    p = _write_yaml(tmp_path, "stokes_vector_robin.yaml", data)

    with pytest.raises(ValueError, match="unsupported operator"):
        load_config(p)


def test_load_config_vector_absorbing_bc(tmp_path):
    data = _base_maxwell_pulse_yaml_dict()
    p = _write_yaml(tmp_path, "maxwell_pulse_absorbing.yaml", data)

    cfg = load_config(p)
    bc = cfg.boundary_field("electric_field").side_conditions("x-")[0]
    assert bc.type == "absorbing"
    assert bc.value.type == "zero"


def test_load_config_resolves_mapping_fragment_ref(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {"$ref": "solver.profile.stationary_scalar_spd"}
    p = _write_yaml(tmp_path, "solver_ref.yaml", data)

    cfg = load_config(p)

    assert cfg.solver.strategy == STATIONARY_SCALAR_SPD
    assert cfg.solver.serial["pc_type"] == "lu"
    assert cfg.solver.mpi["pc_type"] == "hypre"


def test_load_config_resolves_list_and_scalar_fragment_refs(tmp_path):
    data = _base_yaml_dict()
    data["output"] = {
        "resolution": [8, 8],
        "num_frames": 1,
        "formats": {"$ref": "output.formats.numpy_gif"},
        "fields": {"u": {"$ref": "output.mode.scalar"}},
    }
    p = _write_yaml(tmp_path, "output_ref.yaml", data)

    cfg = load_config(p)

    assert cfg.output.formats == ["numpy", "gif"]
    assert cfg.output_mode("u") == "scalar"


def test_load_config_fragment_override_deep_merges_mappings(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {
        "$ref": "solver.profile.stationary_scalar_spd",
        "mpi": {"ksp_rtol": 1.0e-8},
    }
    p = _write_yaml(tmp_path, "solver_override.yaml", data)

    cfg = load_config(p)

    assert cfg.solver.mpi["pc_type"] == "hypre"
    assert float(cfg.solver.mpi["ksp_rtol"]) == pytest.approx(1.0e-8)


def test_load_config_fragment_override_replaces_lists(tmp_path):
    data = _base_yaml_dict()
    data["output"] = {
        "$ref": "output.template.scalar_u",
        "resolution": [8, 8],
        "num_frames": 1,
        "formats": ["vtk"],
    }
    p = _write_yaml(tmp_path, "output_override.yaml", data)

    cfg = load_config(p)

    assert cfg.output.formats == ["vtk"]


def test_load_config_missing_fragment_ref(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {"$ref": "solver.profile.missing"}
    p = _write_yaml(tmp_path, "missing_ref.yaml", data)

    with pytest.raises(ValueError, match="unknown fragment 'solver.profile.missing'"):
        load_config(p)


def test_load_config_fragment_cycle_is_rejected(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {"$ref": "test.cycle.a"}
    p = _write_yaml(tmp_path, "cycle_ref.yaml", data)

    with pytest.raises(ValueError, match="Detected fragment reference cycle"):
        load_config(p)


def test_load_config_rejects_override_on_non_mapping_fragment(tmp_path):
    data = _base_yaml_dict()
    data["output"]["fields"]["u"] = {
        "$ref": "output.mode.scalar",
        "mode": "components",
    }
    p = _write_yaml(tmp_path, "bad_scalar_override.yaml", data)

    with pytest.raises(ValueError, match="cannot apply local overrides"):
        load_config(p)


def test_load_config_rejects_unknown_top_level_keys(tmp_path):
    data = _base_yaml_dict()
    data["unexpected"] = {"$ref": "output.mode.scalar"}
    p = _write_yaml(tmp_path, "bad_top_level.yaml", data)

    with pytest.raises(ValueError, match="unsupported top-level keys"):
        load_config(p)
