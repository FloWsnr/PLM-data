"""Tests for plm_data.core.config."""

import pytest
import yaml

from plm_data.core.config import load_config


def test_load_config():
    cfg = load_config("configs/basic/heat/2d_default.yaml")
    assert cfg.preset == "heat"
    assert "kappa" in cfg.parameters
    assert cfg.domain.type == "rectangle"
    assert cfg.output_resolution == [64, 64]
    assert cfg.dt == 0.01
    assert cfg.t_end == 1.0
    assert "u" in cfg.initial_conditions
    assert cfg.initial_conditions["u"].type == "gaussian_bump"
    assert "u" in cfg.boundary_conditions
    assert "u" in cfg.source_terms


def test_load_config_missing_field(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump({"parameters": {"k": 1.0}}))
    with pytest.raises(ValueError, match="preset"):
        load_config(bad_yaml)


def test_load_config_per_field_boundary_conditions():
    cfg = load_config("configs/basic/poisson/2d_default.yaml")
    assert "u" in cfg.boundary_conditions
    u_bcs = cfg.boundary_conditions["u"]
    assert "x-" in u_bcs
    assert "x+" in u_bcs
    assert "y-" in u_bcs
    assert "y+" in u_bcs
    for bc in u_bcs.values():
        assert bc.type == "dirichlet"


def test_load_config_per_field_source_terms():
    cfg = load_config("configs/basic/poisson/2d_default.yaml")
    assert "u" in cfg.source_terms
    assert cfg.source_terms["u"].type == "sine_product"


def test_load_config_per_field_initial_conditions():
    cfg = load_config("configs/basic/heat/2d_default.yaml")
    assert "u" in cfg.initial_conditions
    assert cfg.initial_conditions["u"].type == "gaussian_bump"


def test_load_config_steady_state_no_ics():
    cfg = load_config("configs/basic/poisson/2d_default.yaml")
    assert cfg.initial_conditions == {}


def _base_yaml_dict():
    """Return a minimal valid config dict with all required top-level fields."""
    return {
        "preset": "test",
        "parameters": {"k": 1.0},
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "boundary_conditions": {
            "u": {
                "x-": {"type": "dirichlet", "value": 0.0},
            },
        },
        "source_terms": {
            "u": {"type": "none"},
        },
        "output_resolution": [8, 8],
        "output": {
            "path": "./output",
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
    data["boundary_conditions"]["u"]["x-"] = {"type": "robin", "value": 0.0}
    p = tmp_path / "robin_no_alpha.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="alpha"):
        load_config(p)


def test_missing_boundary_conditions(tmp_path):
    data = _base_yaml_dict()
    del data["boundary_conditions"]
    p = tmp_path / "no_bcs.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="boundary_conditions"):
        load_config(p)


def test_missing_source_terms(tmp_path):
    data = _base_yaml_dict()
    del data["source_terms"]
    p = tmp_path / "no_src.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="source_terms"):
        load_config(p)
