"""Tests for configuration loading and validation."""

import pytest
import yaml
from pathlib import Path

from pde_sim.core.config import (
    SimulationConfig,
    OutputConfig,
    BoundaryConfig,
    InitialConditionConfig,
    load_config,
    config_to_dict,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, sample_config_file):
        """Test loading a valid configuration file."""
        config = load_config(sample_config_file)

        assert config.preset == "heat"
        assert config.parameters["D"] == 0.1
        assert config.init.type == "gaussian-blobs"
        assert config.solver == "euler"
        assert config.timesteps == 100
        assert config.dt == 0.0001
        assert config.resolution == 32
        assert config.bc.x == "periodic"
        assert config.bc.y == "periodic"
        assert config.seed == 42

    def test_load_config_with_defaults(self, tmp_path):
        """Test that defaults are applied for missing optional fields."""
        minimal_config = {
            "preset": "heat",
            "init": {"type": "random-uniform"},
            "timesteps": 100,
            "dt": 0.001,
            "resolution": 64,
        }

        config_path = tmp_path / "minimal.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config, f)

        config = load_config(config_path)

        # Check defaults
        assert config.parameters == {}
        assert config.solver == "euler"
        assert config.bc.x == "periodic"
        assert config.bc.y == "periodic"
        assert config.seed is None
        assert config.domain_size == 1.0

    def test_load_config_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_missing_required_field(self, tmp_path):
        """Test that KeyError is raised for missing required fields."""
        incomplete_config = {
            "preset": "heat",
            # Missing init, timesteps, dt, resolution
        }

        config_path = tmp_path / "incomplete.yaml"
        with open(config_path, "w") as f:
            yaml.dump(incomplete_config, f)

        with pytest.raises(KeyError):
            load_config(config_path)


class TestConfigToDict:
    """Tests for config_to_dict function."""

    def test_roundtrip_conversion(self, sample_config_file):
        """Test that config can be converted to dict and back."""
        config = load_config(sample_config_file)
        config_dict = config_to_dict(config)

        # Check essential fields
        assert config_dict["preset"] == "heat"
        assert config_dict["parameters"]["D"] == 0.1
        assert config_dict["init"]["type"] == "gaussian-blobs"
        assert config_dict["solver"] == "euler"
        assert config_dict["timesteps"] == 100
        assert config_dict["resolution"] == 32


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_create_config(self):
        """Test creating a SimulationConfig manually."""
        config = SimulationConfig(
            preset="heat",
            parameters={"D": 1.0},
            init=InitialConditionConfig(type="random-uniform", params={}),
            solver="euler",
            timesteps=100,
            dt=0.001,
            resolution=64,
            bc=BoundaryConfig(x="periodic", y="periodic"),
            output=OutputConfig(path=Path("./output")),
        )

        assert config.preset == "heat"
        assert config.resolution == 64
        assert config.seed is None


class TestBoundaryConfig:
    """Tests for BoundaryConfig dataclass."""

    def test_default_values(self):
        """Test that BoundaryConfig has correct defaults."""
        bc = BoundaryConfig()
        assert bc.x == "periodic"
        assert bc.y == "periodic"

    def test_custom_values(self):
        """Test BoundaryConfig with custom values."""
        bc = BoundaryConfig(x="neumann", y="dirichlet")
        assert bc.x == "neumann"
        assert bc.y == "dirichlet"


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_values(self):
        """Test that OutputConfig has correct defaults."""
        output = OutputConfig(path=Path("./test"))
        assert output.frames_per_save == 10
        assert output.colormap == "turbo"
        assert output.field_to_plot is None

    def test_custom_values(self):
        """Test OutputConfig with custom values."""
        output = OutputConfig(
            path=Path("./custom"),
            frames_per_save=50,
            colormap="viridis",
            field_to_plot="u",
        )
        assert output.frames_per_save == 50
        assert output.colormap == "viridis"
        assert output.field_to_plot == "u"
