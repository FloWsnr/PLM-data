"""Smoke-test every YAML config in configs/ to ensure they run without errors.

Configs are discovered automatically. Time-dependent simulations are truncated
to a single timestep. Output resolution is reduced for speed.
"""

from pathlib import Path
import importlib.util

import pytest

from plm_data.core.config import load_config
from plm_data.core.config_realization import realize_simulation_config
from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ALL_CONFIGS = sorted(
    path for path in CONFIGS_DIR.rglob("*.yaml") if not path.name.startswith("_")
)
HAS_DOLFINX_MPC = importlib.util.find_spec("dolfinx_mpc") is not None
HAS_GMSH = importlib.util.find_spec("gmsh") is not None
SMOKE_MESH_RESOLUTION_CAP = 4
GMSH_DOMAIN_TYPES = {
    "annulus",
    "airfoil_channel",
    "channel_obstacle",
    "disk",
    "dumbbell",
    "l_shape",
    "y_bifurcation",
    "venturi_channel",
    "serpentine_channel",
    "side_cavity_channel",
}
SMOKE_MESH_SIZE_FLOOR = 0.3
SMOKE_NARROW_GMSH_CHANNEL_MESH_SIZE_FLOOR = 0.15


def _prepare_smoke_run_config(cfg, output_path: Path) -> None:
    """Keep config coverage broad without paying production-scale solve costs."""

    ndim = cfg.domain.dimension
    cfg.output.resolution = [4] * ndim
    cfg.output.path = output_path
    cfg.output.num_frames = 2

    mesh_resolution = cfg.domain.params.get("mesh_resolution")
    if mesh_resolution is not None:
        cfg.domain.params["mesh_resolution"] = [
            min(int(value), SMOKE_MESH_RESOLUTION_CAP) for value in mesh_resolution
        ]

    mesh_size = cfg.domain.params.get("mesh_size")
    if mesh_size is not None:
        mesh_size_floor = SMOKE_MESH_SIZE_FLOOR
        if cfg.domain.type in {
            "airfoil_channel",
            "channel_obstacle",
            "y_bifurcation",
            "venturi_channel",
            "serpentine_channel",
            "side_cavity_channel",
        }:
            mesh_size_floor = SMOKE_NARROW_GMSH_CHANNEL_MESH_SIZE_FLOOR
        cfg.domain.params["mesh_size"] = max(float(mesh_size), mesh_size_floor)

    if cfg.time is not None:
        cfg.time.t_end = cfg.time.dt  # single timestep


@pytest.mark.parametrize(
    "config_path",
    ALL_CONFIGS,
    ids=[str(p.relative_to(CONFIGS_DIR)) for p in ALL_CONFIGS],
)
def test_config_runs(config_path, tmp_path):
    cfg = load_config(config_path)
    if cfg.has_periodic_boundary_conditions and not HAS_DOLFINX_MPC:
        pytest.skip("periodic configs require dolfinx_mpc")
    if cfg.domain.type in GMSH_DOMAIN_TYPES and not HAS_GMSH:
        pytest.skip("gmsh domain requires python-gmsh")

    # This is a parser/build/solve smoke sweep, not a fidelity benchmark.
    cfg = realize_simulation_config(cfg)
    _prepare_smoke_run_config(cfg, tmp_path)

    preset = get_preset(cfg.preset)
    output_dir = tmp_path / "out"
    writer = FrameWriter(output_dir, cfg, preset.spec)

    result = preset.build_problem(cfg).run(writer)

    assert result.solver_converged is True
    assert writer.frame_count >= 1
