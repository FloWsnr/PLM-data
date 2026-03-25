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
