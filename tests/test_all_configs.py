"""Smoke-test every YAML config in configs/ to ensure they run without errors.

Configs are discovered automatically. Time-dependent simulations are truncated
to a single timestep. Output resolution is reduced for speed.
"""

from pathlib import Path
import importlib.util

import pytest

from plm_data.core.config import load_config
from plm_data.core.output import FrameWriter
from plm_data.core.runtime import is_complex_runtime
from plm_data.presets import get_preset

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ALL_CONFIGS = sorted(CONFIGS_DIR.rglob("*.yaml"))
HAS_DOLFINX_MPC = importlib.util.find_spec("dolfinx_mpc") is not None


@pytest.mark.parametrize(
    "config_path",
    ALL_CONFIGS,
    ids=[str(p.relative_to(CONFIGS_DIR)) for p in ALL_CONFIGS],
)
def test_config_runs(config_path, tmp_path):
    cfg = load_config(config_path)
    if cfg.preset == "maxwell" and not is_complex_runtime():
        pytest.skip("harmonic Maxwell requires a complex-valued runtime")
    if cfg.has_periodic_boundary_conditions and not HAS_DOLFINX_MPC:
        pytest.skip("periodic configs require dolfinx_mpc")

    # Shrink to minimal run (match dimensionality of the domain)
    ndim = cfg.domain.dimension
    cfg.output.resolution = [4] * ndim
    cfg.output.path = tmp_path
    cfg.output.num_frames = 2
    if cfg.time is not None:
        cfg.time.t_end = cfg.time.dt  # single timestep

    preset = get_preset(cfg.preset)
    output_dir = tmp_path / "out"
    writer = FrameWriter(output_dir, cfg, preset.spec)

    result = preset.build_problem(cfg).run(writer)

    assert result.solver_converged is True
    assert writer.frame_count >= 1
