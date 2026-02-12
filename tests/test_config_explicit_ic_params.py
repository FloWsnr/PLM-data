"""Ensure configs do not rely on hidden IC defaults."""

from __future__ import annotations

import copy
import inspect
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from pde_sim.boundaries import create_grid_with_bc
from pde_sim.core.config import load_config
from pde_sim.initial_conditions import _IC_REGISTRY
from pde_sim.pdes import get_pde_preset


class _TrackingDict(dict):
    """Dict that records missing-key accesses with non-None defaults."""

    def __init__(
        self,
        *args,
        _missing_defaults: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._missing_defaults = _missing_defaults if _missing_defaults is not None else {}

    def get(self, key, default=None):
        if key not in self and default is not None and key not in self._missing_defaults:
            self._missing_defaults[key] = default
        return super().get(key, default)


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
        if name in {"self", "grid", "kwargs", "seed"}:
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
def test_configs_have_explicit_ic_params(config_path: Path):
    """Configs should not rely on hidden defaults in IC generators/presets."""
    raw = yaml.safe_load(config_path.read_text()) or {}
    init = raw.get("init", {})
    if not isinstance(init, dict):
        return
    params = init.get("params", {})
    if not isinstance(params, dict):
        return

    ic_type = init.get("type")

    # 1) Generic IC defaults (function signature defaults) should be explicit.
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

    # 2) Custom preset IC defaults (ic_params.get(..., default)) should be explicit.
    cfg = load_config(config_path)
    preset = get_pde_preset(cfg.preset)
    grid = create_grid_with_bc(cfg.resolution, cfg.domain_size, cfg.bc)

    runtime_ic_params = copy.deepcopy(cfg.init.params)
    if cfg.seed is not None and "seed" not in runtime_ic_params:
        runtime_ic_params["seed"] = cfg.seed

    missing_resolve: dict[str, Any] = {}
    tracked_resolve = _TrackingDict(runtime_ic_params, _missing_defaults=missing_resolve)
    resolved = preset.resolve_ic_params(
        grid=grid,
        ic_type=cfg.init.type,
        ic_params=tracked_resolve,
    )

    missing_create: dict[str, Any] = {}
    tracked_create = _TrackingDict(resolved, _missing_defaults=missing_create)
    preset.create_initial_state(
        grid=grid,
        ic_type=cfg.init.type,
        ic_params=tracked_create,
        parameters=cfg.parameters,
        bc=cfg.bc,
    )

    runtime_missing = {**missing_resolve, **missing_create}
    runtime_missing.pop("seed", None)
    unresolved = [name for name in runtime_missing if name not in params]
    assert not unresolved, (
        f"{config_path}: missing explicit custom IC params for '{cfg.preset}/{cfg.init.type}': "
        f"{sorted(unresolved)}"
    )

