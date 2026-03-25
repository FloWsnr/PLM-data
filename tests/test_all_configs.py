"""Smoke-test every YAML config in configs/ to ensure they run without errors.

Configs are discovered automatically. Time-dependent simulations are truncated
to a single timestep. Output resolution is reduced for speed.
"""

from pathlib import Path

import pytest

from plm_data.core.config import load_config
from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ALL_CONFIGS = sorted(CONFIGS_DIR.rglob("*.yaml"))


@pytest.mark.parametrize(
    "config_path",
    ALL_CONFIGS,
    ids=[str(p.relative_to(CONFIGS_DIR)) for p in ALL_CONFIGS],
)
def test_config_runs(config_path, tmp_path):
    cfg = load_config(config_path)

    # Shrink to minimal run (match dimensionality of the domain)
    ndim = len(cfg.domain.params.get("size", [1, 1]))
    cfg.output_resolution = [4] * ndim
    cfg.output.path = tmp_path
    cfg.output.num_frames = 2
    if cfg.t_end is not None and cfg.dt is not None:
        cfg.t_end = cfg.dt  # single timestep

    preset = get_preset(cfg.preset)
    output_dir = tmp_path / "out"
    writer = FrameWriter(output_dir, cfg)

    result = preset.run(cfg, writer)

    assert result.solver_converged is True
    assert writer.frame_count >= 1
