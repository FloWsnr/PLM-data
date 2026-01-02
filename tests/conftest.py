"""Pytest configuration and fixtures."""

import pytest
import yaml
from pde import CartesianGrid


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
        "parameters": {"D": 0.1},
        "init": {
            "type": "gaussian-blobs",
            "params": {"num_blobs": 2, "amplitude": 1.0},
        },
        "solver": "euler",
        "t_end": 0.01,  # 100 * 0.0001
        "dt": 0.0001,
        "resolution": 32,
        "bc": {"x": "periodic", "y": "periodic"},
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
