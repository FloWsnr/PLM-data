"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from pde_sim.core.config import (
    SimulationConfig,
    OutputConfig,
    BoundaryConfig,
    InitialConditionConfig,
    load_config,
    config_to_dict,
    _deep_merge,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, sample_config_file):
        """Test loading a valid configuration file."""
        config = load_config(sample_config_file)

        assert config.preset == "heat"
        assert config.parameters["D_T"] == 0.1
        assert config.init.type == "gaussian-blobs"
        assert config.solver == "euler"
        assert config.t_end == 0.01  # 100 * 0.0001
        assert config.dt == 0.0001
        # Resolution is now always a list
        assert config.resolution == [32, 32]
        assert config.ndim == 2
        assert config.bc.x_minus == "periodic"
        assert config.bc.y_minus == "periodic"
        assert config.seed == 42

    def test_load_config_with_defaults(self, tmp_path):
        """Test that defaults are applied for missing optional fields."""
        minimal_config = {
            "preset": "heat",
            "init": {"type": "random-uniform"},
            "t_end": 0.1,  # 100 * 0.001
            "dt": 0.001,
            "resolution": [64, 64],
        }

        config_path = tmp_path / "minimal.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config, f)

        config = load_config(config_path)

        # Check defaults
        assert config.parameters == {}
        assert config.solver == "euler"
        assert config.bc.x_minus == "periodic"
        assert config.bc.y_minus == "periodic"  # Default for 2D
        assert config.seed is None
        assert config.domain_size == [1.0, 1.0]  # Now a list
        assert config.resolution == [64, 64]

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
        assert config_dict["parameters"]["D_T"] == 0.1
        assert config_dict["init"]["type"] == "gaussian-blobs"
        assert config_dict["solver"] == "euler"
        assert config_dict["t_end"] == 0.01  # 100 * 0.0001
        assert config_dict["resolution"] == [32, 32]  # Now a list


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
            resolution=[64, 64],  # Now a list
            bc=BoundaryConfig(
                x_minus="periodic",
                x_plus="periodic",
                y_minus="periodic",
                y_plus="periodic",
            ),
            output=OutputConfig(path=Path("./output")),
        )

        assert config.preset == "heat"
        assert config.resolution == [64, 64]
        assert config.ndim == 2
        assert config.seed is None


class TestBoundaryConfig:
    """Tests for BoundaryConfig dataclass."""

    def test_default_values(self):
        """Test that BoundaryConfig has correct defaults.

        Note: y and z boundaries default to None because they're only
        required for 2D+ and 3D simulations respectively.
        """
        bc = BoundaryConfig()
        assert bc.x_minus == "periodic"
        assert bc.x_plus == "periodic"
        assert bc.y_minus is None  # Not required for 1D
        assert bc.y_plus is None
        assert bc.z_minus is None  # Not required for 1D/2D
        assert bc.z_plus is None

    def test_custom_values(self):
        """Test BoundaryConfig with custom values."""
        bc = BoundaryConfig(
            x_minus="neumann:0",
            x_plus="neumann:0",
            y_minus="dirichlet:0",
            y_plus="dirichlet:0",
        )
        assert bc.x_minus == "neumann:0"
        assert bc.y_minus == "dirichlet:0"

    def test_is_simple_without_fields(self):
        """Test is_simple returns True when no per-field BCs."""
        bc = BoundaryConfig(
            x_minus="periodic",
            x_plus="periodic",
            y_minus="neumann:0",
            y_plus="neumann:0",
        )
        assert bc.is_simple() is True

    def test_is_simple_with_fields(self):
        """Test is_simple returns False when per-field BCs present."""
        bc = BoundaryConfig(
            x_minus="periodic",
            x_plus="periodic",
            y_minus="periodic",
            y_plus="periodic",
            fields={"u": {"x-": "dirichlet:0"}},
        )
        assert bc.is_simple() is False

    def test_get_field_bc_with_override(self):
        """Test get_field_bc returns field-specific BC merged with defaults."""
        bc = BoundaryConfig(
            x_minus="periodic",
            x_plus="periodic",
            y_minus="neumann:0",
            y_plus="neumann:0",
            fields={"omega": {"y-": "dirichlet:0", "y+": "dirichlet:0"}},
        )
        field_bc = bc.get_field_bc("omega")
        assert field_bc["x-"] == "periodic"  # From default
        assert field_bc["x+"] == "periodic"  # From default
        assert field_bc["y-"] == "dirichlet:0"  # Override
        assert field_bc["y+"] == "dirichlet:0"  # Override

    def test_get_field_bc_without_override(self):
        """Test get_field_bc returns defaults for unspecified field."""
        bc = BoundaryConfig(
            x_minus="neumann:0",
            x_plus="neumann:0",
            y_minus="dirichlet:0",
            y_plus="dirichlet:0",
        )
        field_bc = bc.get_field_bc("v")
        assert field_bc["x-"] == "neumann:0"
        assert field_bc["x+"] == "neumann:0"
        assert field_bc["y-"] == "dirichlet:0"
        assert field_bc["y+"] == "dirichlet:0"


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
            "resolution": [32, 32],
            "bc": {
                "x-": "periodic",
                "x+": "periodic",
                "y-": "neumann:0",
                "y+": "neumann:0",
                "fields": {
                    "omega": {"y-": "dirichlet:0", "y+": "dirichlet:0"},
                    "b": {"y-": "neumann:0.08", "y+": "dirichlet:0"},
                },
            },
        }

        config_path = tmp_path / "per_field_bc.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        assert config.bc.x_minus == "periodic"
        assert config.bc.y_minus == "neumann:0"
        assert config.bc.fields is not None
        assert "omega" in config.bc.fields
        assert "b" in config.bc.fields
        assert config.bc.fields["omega"]["y-"] == "dirichlet:0"
        assert config.bc.fields["b"]["y-"] == "neumann:0.08"

    def test_config_to_dict_with_per_field_bc(self, tmp_path):
        """Test config_to_dict preserves per-field BCs."""
        config_data = {
            "preset": "thermal-convection",
            "parameters": {"nu": 0.2},
            "init": {"type": "default"},
            "t_end": 0.1,
            "dt": 0.001,
            "resolution": [32, 32],
            "bc": {
                "x-": "periodic",
                "x+": "periodic",
                "y-": "neumann:0",
                "y+": "neumann:0",
                "fields": {
                    "b": {"y-": "neumann:0.08", "y+": "dirichlet:0"},
                },
            },
        }

        config_path = tmp_path / "per_field_bc.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        result = config_to_dict(config)

        assert "fields" in result["bc"]
        assert "b" in result["bc"]["fields"]
        assert result["bc"]["fields"]["b"]["y-"] == "neumann:0.08"
        assert result["bc"]["fields"]["b"]["y+"] == "dirichlet:0"


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_values(self):
        """Test that OutputConfig has correct defaults."""
        output = OutputConfig(path=Path("./test"))
        assert output.num_frames == 100
        assert output.format == "png"
        assert output.fps == 30

    def test_custom_values(self):
        """Test OutputConfig with custom values."""
        output = OutputConfig(
            path=Path("./custom"),
            num_frames=50,
            format="mp4",
            fps=60,
        )
        assert output.num_frames == 50
        assert output.format == "mp4"
        assert output.fps == 60


class TestDeepMerge:
    """Tests for _deep_merge function."""

    def test_simple_merge(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        base = {"output": {"format": "png", "num_frames": 100}, "seed": 42}
        override = {"output": {"num_frames": 10}, "preset": "heat"}
        result = _deep_merge(base, override)
        assert result["output"]["format"] == "png"  # From base
        assert result["output"]["num_frames"] == 10  # From override
        assert result["seed"] == 42  # From base
        assert result["preset"] == "heat"  # From override

    def test_override_takes_precedence(self):
        """Test that override values always take precedence."""
        base = {"seed": 42}
        override = {"seed": 123}
        result = _deep_merge(base, override)
        assert result["seed"] == 123

    def test_base_not_modified(self):
        """Test that original base dict is not modified."""
        base = {"a": 1, "nested": {"b": 2}}
        override = {"a": 10, "nested": {"c": 3}}
        _deep_merge(base, override)
        assert base == {"a": 1, "nested": {"b": 2}}


class TestMasterConfig:
    """Tests for master config loading and merging."""

    def test_master_config_provides_defaults(self, tmp_path):
        """Test that master config values are used as defaults."""
        # Create master config
        master_config = {
            "output": {"format": "mp4", "num_frames": 200, "fps": 60},
            "seed": 999,
        }
        master_path = tmp_path / "master.yaml"
        with open(master_path, "w") as f:
            yaml.dump(master_config, f)

        # Create individual config without output or seed
        individual_config = {
            "preset": "heat",
            "init": {"type": "random-uniform"},
            "t_end": 0.1,
            "dt": 0.001,
            "resolution": [64, 64],
        }
        config_path = tmp_path / "individual.yaml"
        with open(config_path, "w") as f:
            yaml.dump(individual_config, f)

        config = load_config(config_path)

        # Master config values should be applied
        assert config.output.format == "mp4"
        assert config.output.num_frames == 200
        assert config.output.fps == 60
        assert config.seed == 999

    def test_individual_config_overrides_master(self, tmp_path):
        """Test that individual config values override master config."""
        # Create master config
        master_config = {
            "output": {"format": "mp4", "num_frames": 200},
            "seed": 999,
        }
        master_path = tmp_path / "master.yaml"
        with open(master_path, "w") as f:
            yaml.dump(master_config, f)

        # Create individual config with different values
        individual_config = {
            "preset": "heat",
            "init": {"type": "random-uniform"},
            "t_end": 0.1,
            "dt": 0.001,
            "resolution": [64, 64],
            "output": {"num_frames": 50},  # Override just num_frames
            "seed": 123,  # Override seed
        }
        config_path = tmp_path / "individual.yaml"
        with open(config_path, "w") as f:
            yaml.dump(individual_config, f)

        config = load_config(config_path)

        # Individual values should override master
        assert config.output.format == "mp4"  # From master
        assert config.output.num_frames == 50  # From individual (override)
        assert config.seed == 123  # From individual (override)

    def test_no_master_config(self, tmp_path):
        """Test that configs work without master config."""
        # Create individual config without master
        individual_config = {
            "preset": "heat",
            "init": {"type": "random-uniform"},
            "t_end": 0.1,
            "dt": 0.001,
            "resolution": [64, 64],
        }
        config_path = tmp_path / "individual.yaml"
        with open(config_path, "w") as f:
            yaml.dump(individual_config, f)

        config = load_config(config_path)

        # Should use default values
        assert config.output.format == "png"  # Default
        assert config.output.num_frames == 100  # Default
        assert config.seed is None  # Default

    def test_master_config_in_parent_directory(self, tmp_path):
        """Test that master config is found in parent directory."""
        # Create nested directory structure
        nested_dir = tmp_path / "configs" / "biology" / "test"
        nested_dir.mkdir(parents=True)

        # Create master config in configs/
        master_config = {"seed": 777}
        master_path = tmp_path / "configs" / "master.yaml"
        with open(master_path, "w") as f:
            yaml.dump(master_config, f)

        # Create individual config in nested directory
        individual_config = {
            "preset": "heat",
            "init": {"type": "random-uniform"},
            "t_end": 0.1,
            "dt": 0.001,
            "resolution": [64, 64],
        }
        config_path = nested_dir / "individual.yaml"
        with open(config_path, "w") as f:
            yaml.dump(individual_config, f)

        config = load_config(config_path)

        # Master config should be found and applied
        assert config.seed == 777
