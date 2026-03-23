"""Tests for plm_data.core.config."""

from pathlib import Path

import pytest

from plm_data.core.config import load_config


def test_load_config():
    cfg = load_config("configs/basic/heat/2d_default.yaml")
    assert cfg.preset == "heat"
    assert "kappa" in cfg.parameters
    assert cfg.domain.type == "rectangle"
    assert cfg.output_resolution == [64, 64]
    assert cfg.dt == 0.01
    assert cfg.t_end == 1.0
    assert cfg.initial_condition is not None
    assert cfg.initial_condition.type == "gaussian_bump"


def test_load_config_missing_field(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump({"parameters": {"k": 1.0}}))
    with pytest.raises(ValueError, match="preset"):
        load_config(bad_yaml)
