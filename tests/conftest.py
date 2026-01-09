"""Pytest configuration and fixtures."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset
from pde_sim.core.config import BoundaryConfig


# Path to configs
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(preset_name: str, category: str) -> dict:
    """Load a default config YAML for a given preset.

    Args:
        preset_name: Name of the preset (e.g., "heat", "gray-scott")
        category: Category folder (e.g., "basic", "physics", "biology", "fluids")

    Returns:
        Dictionary with the config contents
    """
    # Try different file naming conventions
    # New structure: configs/{category}/{pde_name}/default.yaml
    possible_names = [
        preset_name,
        preset_name.replace("-", "_"),
    ]

    for name in possible_names:
        config_path = CONFIGS_DIR / category / name / "default.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Config not found for {preset_name} in {category}. "
        f"Tried: {[CONFIGS_DIR / category / n / 'default.yaml' for n in possible_names]}"
    )


def run_short_simulation(
    preset_name: str,
    category: str,
    t_end: float = 0.01,
    resolution: int = 32,
    seed: int = 42,
) -> tuple:
    """Run a short simulation using the default config but with reduced t_end and resolution.

    Args:
        preset_name: Name of the preset (e.g., "heat", "gray-scott")
        category: Category folder (e.g., "basic", "physics", "biology", "fluids")
        t_end: End time for the simulation (default 0.01 for speed)
        resolution: Grid resolution (default 32 for speed)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (result, config) where result is the final state
    """
    np.random.seed(seed)

    # Load config
    config = load_config(preset_name, category)

    # Get preset
    pde_preset = get_pde_preset(preset_name)

    # Create grid with reduced resolution
    domain_size = config.get("domain_size", 1.0)
    bc_raw = config.get("bc", {})

    # Parse BC config to determine periodicity
    bc_config = BoundaryConfig(
        x_minus=bc_raw.get("x-", "periodic"),
        x_plus=bc_raw.get("x+", "periodic"),
        y_minus=bc_raw.get("y-", "periodic"),
        y_plus=bc_raw.get("y+", "periodic"),
        fields=bc_raw.get("fields"),
    )

    # Determine if grid should be periodic based on BC
    periodic_x = bc_config.x_minus == "periodic" and bc_config.x_plus == "periodic"
    periodic_y = bc_config.y_minus == "periodic" and bc_config.y_plus == "periodic"

    grid = CartesianGrid(
        bounds=[[0, domain_size], [0, domain_size]],
        shape=[resolution, resolution],
        periodic=[periodic_x, periodic_y],
    )

    # Get parameters from config
    params = config.get("parameters", pde_preset.get_default_parameters())

    # Create PDE - pass bc_config directly
    pde = pde_preset.create_pde(
        parameters=params,
        bc=bc_config,
        grid=grid,
    )

    # Create initial state
    init_config = config.get("init", {"type": "default", "params": {}})
    ic_type = init_config.get("type", "default")
    ic_params = init_config.get("params", {})

    state = pde_preset.create_initial_state(
        grid=grid,
        ic_type=ic_type,
        ic_params=ic_params,
    )

    # Get solver settings from config
    solver = config.get("solver", "euler")
    # Map common solver aliases to py-pde solver names
    solver_map = {
        "rk4": "runge-kutta",
        "rk": "runge-kutta",
        "midpoint": "runge-kutta",  # midpoint is a 2nd-order RK method
    }
    solver = solver_map.get(solver, solver)
    dt = config.get("dt", 0.001)

    # Adjust dt if needed for the short simulation
    # Make sure we don't have more steps than needed
    num_steps = t_end / dt
    if num_steps < 1:
        dt = t_end / 2  # At least 2 steps

    # Run simulation
    result = pde.solve(
        state,
        t_range=t_end,
        dt=dt,
        solver=solver,
        tracker=None,
    )

    return result, config


@pytest.fixture
def config_simulation_runner():
    """Fixture that returns the run_short_simulation function."""
    return run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small 32x32 grid for fast testing."""
    return CartesianGrid(
        bounds=[[0, 1], [0, 1]],
        shape=[32, 32],
        periodic=[True, True],
    )


@pytest.fixture
def medium_grid():
    """Create a medium 64x64 grid."""
    return CartesianGrid(
        bounds=[[0, 1], [0, 1]],
        shape=[64, 64],
        periodic=[True, True],
    )


@pytest.fixture
def non_periodic_grid():
    """Create a grid with non-periodic boundary conditions."""
    return CartesianGrid(
        bounds=[[0, 1], [0, 1]],
        shape=[32, 32],
        periodic=[False, False],
    )


@pytest.fixture
def tmp_output(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_config_dict():
    """Return a sample configuration dictionary."""
    return {
        "preset": "heat",
        "parameters": {"D_T": 0.1},
        "init": {
            "type": "gaussian-blobs",
            "params": {"num_blobs": 2, "amplitude": 1.0},
        },
        "solver": "euler",
        "t_end": 0.01,  # 100 * 0.0001
        "dt": 0.0001,
        "resolution": 32,
        "bc": {
            "x-": "periodic",
            "x+": "periodic",
            "y-": "periodic",
            "y+": "periodic",
        },
        "output": {
            "path": "./output",
            "num_frames": 10,
            "colormap": "turbo",
        },
        "seed": 42,
        "domain_size": 1.0,
    }


@pytest.fixture
def sample_config_file(tmp_path, sample_config_dict):
    """Create a sample config YAML file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path
