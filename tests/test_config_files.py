"""Validate every YAML config by instantiating SimulationRunner (no time stepping)."""

from pathlib import Path

import pytest

from pde_sim.core.config import load_config
from pde_sim.core.simulation import SimulationRunner

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def collect_config_files() -> list[Path]:
    return sorted(p for p in CONFIGS_DIR.rglob("*.yaml") if p.name != "master.yaml")


def config_id(path: Path) -> str:
    return str(path.relative_to(CONFIGS_DIR).with_suffix(""))


@pytest.mark.parametrize(
    "config_path",
    collect_config_files(),
    ids=[config_id(p) for p in collect_config_files()],
)
def test_config_valid(config_path: Path, tmp_path: Path) -> None:
    """Validate config by running the full setup pipeline (0 time steps)."""
    config = load_config(config_path)
    SimulationRunner(config, output_dir=tmp_path)
