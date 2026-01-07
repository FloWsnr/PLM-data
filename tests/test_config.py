"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from pde_sim.core.config import (
    SimulationConfig,
    OutputConfig,
    BoundaryConfig,
    FieldBoundaryConfig,
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
        assert config.t_end == 0.01  # 100 * 0.0001
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
            "t_end": 0.1,  # 100 * 0.001
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
            # Missing init, t_end, dt, resolution
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
        assert config_dict["t_end"] == 0.01  # 100 * 0.0001
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
            t_end=0.1,  # 100 * 0.001
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

    def test_is_simple_without_fields(self):
        """Test is_simple returns True when no per-field BCs."""
        bc = BoundaryConfig(x="periodic", y="neumann")
        assert bc.is_simple() is True

    def test_is_simple_with_fields(self):
        """Test is_simple returns False when per-field BCs present."""
        bc = BoundaryConfig(
            x="periodic",
            y="periodic",
            fields={"u": FieldBoundaryConfig(x="dirichlet")}
        )
        assert bc.is_simple() is False

    def test_get_field_bc_with_override(self):
        """Test get_field_bc returns field-specific BC."""
        bc = BoundaryConfig(
            x="periodic",
            y="periodic",
            fields={"u": FieldBoundaryConfig(x="dirichlet", top="neumann")}
        )
        field_bc = bc.get_field_bc("u")
        assert field_bc.x == "dirichlet"
        assert field_bc.y == "periodic"  # Falls back to default
        assert field_bc.top == "neumann"

    def test_get_field_bc_without_override(self):
        """Test get_field_bc returns defaults for unspecified field."""
        bc = BoundaryConfig(x="neumann", y="dirichlet")
        field_bc = bc.get_field_bc("v")
        assert field_bc.x == "neumann"
        assert field_bc.y == "dirichlet"


class TestFieldBoundaryConfig:
    """Tests for FieldBoundaryConfig dataclass."""

    def test_default_values(self):
        """Test FieldBoundaryConfig defaults to None."""
        fbc = FieldBoundaryConfig()
        assert fbc.x is None
        assert fbc.y is None
        assert fbc.left is None
        assert fbc.right is None
        assert fbc.top is None
        assert fbc.bottom is None

    def test_side_specific_values(self):
        """Test FieldBoundaryConfig with side-specific values."""
        fbc = FieldBoundaryConfig(
            x="periodic",
            top="dirichlet:0",
            bottom="neumann:0.08"
        )
        assert fbc.x == "periodic"
        assert fbc.top == "dirichlet:0"
        assert fbc.bottom == "neumann:0.08"


class TestPerFieldBCConfigParsing:
    """Tests for loading configs with per-field boundary conditions."""

    def test_load_per_field_bc_config(self, tmp_path):
        """Test loading config with per-field BCs."""
        config_data = {
            "preset": "thermal-convection",
            "parameters": {"nu": 0.2},
            "init": {"type": "default"},
            "t_end": 0.1,
            "dt": 0.001,
            "resolution": 32,
            "bc": {
                "x": "periodic",
                "y": "periodic",
                "fields": {
                    "omega": {"top": "dirichlet:0", "bottom": "dirichlet:0"},
                    "b": {"top": "dirichlet:0", "bottom": "neumann:0.08"}
                }
            }
        }

        config_path = tmp_path / "per_field_bc.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        assert config.bc.x == "periodic"
        assert config.bc.y == "periodic"
        assert config.bc.fields is not None
        assert "omega" in config.bc.fields
        assert "b" in config.bc.fields
        assert config.bc.fields["omega"].top == "dirichlet:0"
        assert config.bc.fields["b"].bottom == "neumann:0.08"

    def test_config_to_dict_with_per_field_bc(self, tmp_path):
        """Test config_to_dict preserves per-field BCs."""
        config_data = {
            "preset": "thermal-convection",
            "parameters": {"nu": 0.2},
            "init": {"type": "default"},
            "t_end": 0.1,
            "dt": 0.001,
            "resolution": 32,
            "bc": {
                "x": "periodic",
                "y": "periodic",
                "fields": {
                    "b": {"top": "dirichlet:0", "bottom": "neumann:0.08"}
                }
            }
        }

        config_path = tmp_path / "per_field_bc.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        result = config_to_dict(config)

        assert "fields" in result["bc"]
        assert "b" in result["bc"]["fields"]
        assert result["bc"]["fields"]["b"]["top"] == "dirichlet:0"
        assert result["bc"]["fields"]["b"]["bottom"] == "neumann:0.08"


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_values(self):
        """Test that OutputConfig has correct defaults."""
        output = OutputConfig(path=Path("./test"))
        assert output.num_frames == 100
        assert output.colormap == "turbo"
        assert output.field_to_plot is None

    def test_custom_values(self):
        """Test OutputConfig with custom values."""
        output = OutputConfig(
            path=Path("./custom"),
            num_frames=50,
            colormap="viridis",
            field_to_plot="u",
        )
        assert output.num_frames == 50
        assert output.colormap == "viridis"
        assert output.field_to_plot == "u"
