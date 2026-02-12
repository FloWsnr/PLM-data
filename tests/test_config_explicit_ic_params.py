"""Validate every tracked YAML config: setup pipeline and no hidden IC defaults."""

from __future__ import annotations

import copy
import inspect
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from pde_sim.core.config import load_config
from pde_sim.core.simulation import SimulationRunner
from pde_sim.initial_conditions import _IC_REGISTRY


def _normalize(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.floating, np.integer)):
            return value.item()
    except Exception:
        pass

    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _tracked_config_paths() -> list[Path]:
    out = subprocess.check_output(
        ["bash", "-lc", "git ls-files 'configs/**/*.yaml'"]
    ).decode()
    return [
        Path(line)
        for line in out.splitlines()
        if line and not line.endswith("master.yaml")
    ]


def _generic_ic_defaults(ic_type: str) -> dict[str, Any]:
    if ic_type not in _IC_REGISTRY:
        return {}

    signature = inspect.signature(_IC_REGISTRY[ic_type].generate)
    defaults: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if name in {"self", "grid", "kwargs", "seed", "randomize"}:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect._empty or param.default is None:
            continue
        defaults[name] = _normalize(copy.deepcopy(param.default))
    return defaults


@pytest.mark.parametrize("config_path", _tracked_config_paths())
def test_config_valid(config_path: Path, tmp_path: Path):
    """Validate config by running the full setup pipeline (no time stepping)."""
    cfg = load_config(config_path)
    SimulationRunner(cfg, output_dir=tmp_path)


@pytest.mark.parametrize("config_path", _tracked_config_paths())
def test_configs_have_explicit_ic_params(config_path: Path):
    """Configs should not rely on hidden defaults in IC generators."""
    raw = yaml.safe_load(config_path.read_text()) or {}
    init = raw.get("init", {})
    if not isinstance(init, dict):
        return
    params = init.get("params", {})
    if not isinstance(params, dict):
        return

    ic_type = init.get("type")

    # Generic IC defaults (function signature defaults) should be explicit.
    if isinstance(ic_type, str):
        missing_generic = [
            name for name in _generic_ic_defaults(ic_type) if name not in params
        ]
        assert not missing_generic, (
            f"{config_path}: missing generic IC params for '{ic_type}': "
            f"{sorted(missing_generic)}"
        )

    # Check generic defaults for per-field IC overrides as well.
    for _, field_spec in params.items():
        if not isinstance(field_spec, dict):
            continue
        field_type = field_spec.get("type", ic_type)
        field_params = field_spec.get("params", {})
        if not isinstance(field_params, dict) or not isinstance(field_type, str):
            continue

        missing_field_generic = [
            name
            for name in _generic_ic_defaults(field_type)
            if name not in field_params
        ]
        assert not missing_field_generic, (
            f"{config_path}: missing per-field IC params for '{field_type}': "
            f"{sorted(missing_field_generic)}"
        )

