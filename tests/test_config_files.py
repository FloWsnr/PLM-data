"""Validate every YAML config file under configs/ against structural rules.

Catches bugs like missing IC position params, invalid BC strings, unsupported
dimensions, and missing PDE parameters before any simulation runs.
"""

from pathlib import Path

import pytest

from pde_sim.boundaries.factory import BoundaryConditionFactory
from pde_sim.core.config import load_config
from pde_sim.initial_conditions import get_ic_position_params, list_initial_conditions
from pde_sim.pdes import get_pde_preset

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Dimension-aware phase params for sine/cosine IC types
_PHASE_PARAMS_BY_NDIM = {
    1: {"phase_x"},
    2: {"phase_x", "phase_y"},
    3: {"phase_x", "phase_y", "phase_z"},
}


def collect_config_files() -> list[Path]:
    """Glob all YAML configs under configs/, excluding master.yaml."""
    return sorted(
        p for p in CONFIGS_DIR.rglob("*.yaml") if p.name != "master.yaml"
    )


def config_id(path: Path) -> str:
    """Derive a readable test ID like 'basic/heat/default' from full path."""
    rel = path.relative_to(CONFIGS_DIR)
    return str(rel.with_suffix(""))


def get_required_position_params(ic_type: str, ndim: int) -> set[str]:
    """Return required position params for an IC type, filtered by ndim.

    For sine/cosine, only require phase params matching the dimensionality.
    All other IC types return their full get_ic_position_params() set.
    """
    all_params = get_ic_position_params(ic_type)
    if ic_type in ("sine", "cosine"):
        return _PHASE_PARAMS_BY_NDIM[ndim]
    return all_params


def _validate_bc_string(bc_str: str) -> None:
    """Validate a single BC format string via BoundaryConditionFactory.convert()."""
    BoundaryConditionFactory.convert(bc_str)


def _validate_ic_position_params(ic_type: str, ic_params: dict, ndim: int) -> None:
    """Check that required position params are present in ic_params."""
    registered = list_initial_conditions()
    if ic_type not in registered:
        # PDE-specific IC type (default, custom, vortex-pair, etc.) — skip
        return
    required = get_required_position_params(ic_type, ndim)
    missing = required - set(ic_params.keys())
    if missing:
        raise AssertionError(
            f"IC type '{ic_type}' missing position params: {sorted(missing)}"
        )


@pytest.mark.parametrize(
    "config_path",
    collect_config_files(),
    ids=[config_id(p) for p in collect_config_files()],
)
def test_config_valid(config_path: Path) -> None:
    """Validate a single config file against all structural rules."""
    # 1. Config loads (valid YAML, required fields, master merge)
    config = load_config(config_path)

    # 2. Preset exists
    preset = get_pde_preset(config.preset)

    # 3. Dimension supported
    preset.validate_dimension(config.ndim)

    # 4. Required PDE params present
    required_params = {p.name for p in preset.metadata.parameters}
    provided_params = set(config.parameters.keys())
    missing_params = required_params - provided_params
    assert not missing_params, (
        f"Missing PDE parameters: {sorted(missing_params)}"
    )

    # 5. BCs valid for ndim
    config.bc.validate_for_ndim(config.ndim)

    # 6. BC format strings valid
    _validate_bc_string(config.bc.x_minus)
    _validate_bc_string(config.bc.x_plus)
    if config.ndim >= 2:
        _validate_bc_string(config.bc.y_minus)
        _validate_bc_string(config.bc.y_plus)
    if config.ndim >= 3:
        _validate_bc_string(config.bc.z_minus)
        _validate_bc_string(config.bc.z_plus)
    # Per-field BC overrides
    if config.bc.fields:
        for field_name, field_bcs in config.bc.fields.items():
            for side, bc_str in field_bcs.items():
                _validate_bc_string(bc_str)

    # 7. IC position params present
    ic_type = config.init.type
    ic_params = config.init.params
    field_names = preset.metadata.field_names

    # Check for per-field IC overrides
    has_per_field = any(
        name in ic_params and isinstance(ic_params[name], dict)
        for name in field_names
    )

    if has_per_field:
        # Validate each field's IC separately
        for name in field_names:
            if name in ic_params and isinstance(ic_params[name], dict):
                field_spec = ic_params[name]
                field_ic_type = field_spec.get("type", ic_type)
                field_ic_params = field_spec.get("params", {})
                _validate_ic_position_params(field_ic_type, field_ic_params, config.ndim)
            else:
                # Field uses the global IC type
                _validate_ic_position_params(ic_type, ic_params, config.ndim)
    else:
        _validate_ic_position_params(ic_type, ic_params, config.ndim)
