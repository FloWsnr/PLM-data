"""Tests for simulation runner."""

import warnings

import numpy as np
import pytest
import yaml

from pde_sim.core.config import load_config
from pde_sim.core.simulation import SimulationRunner, run_from_config


class TestSimulationRunner:
    """Tests for SimulationRunner class."""

    def test_initialization(self, sample_config_file, tmp_output):
        """Test SimulationRunner initialization."""
        config = load_config(sample_config_file)
        runner = SimulationRunner(config, output_dir=tmp_output)

        assert runner.config == config
        assert runner.output_dir == tmp_output
        assert runner.preset is not None
        assert runner.grid is not None
        assert runner.pde is not None
        assert runner.state is not None

    def test_initialization_with_seed(self, sample_config_file, tmp_output):
        """Test that seed produces reproducible results."""
        config = load_config(sample_config_file)

        # Run twice with same seed
        runner1 = SimulationRunner(config, output_dir=tmp_output, sim_id="test1")
        state1 = runner1.state.data.copy()

        runner2 = SimulationRunner(config, output_dir=tmp_output, sim_id="test2")
        state2 = runner2.state.data.copy()

        # Initial states should be identical
        np.testing.assert_array_equal(state1, state2)

    def test_run_short_simulation(self, tmp_path):
        """Test running a complete short simulation."""
        # Create config for a very short simulation
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},
            "init": {"type": "random-uniform", "params": {"low": 0.0, "high": 1.0}},
            "solver": "euler",
            "backend": "numpy",
            "t_end": 0.005,  # 50 * 0.0001
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 6, "formats": ["png"]},
            "seed": 42,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        metadata = runner.run(verbose=False)

        # Check metadata
        assert "id" in metadata
        assert metadata["preset"] == "heat"
        assert metadata["simulation"]["totalFrames"] > 0

        # Check output files exist (folder is now named PDEname_date)
        output_dir = tmp_path / metadata["folder_name"]
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "frames").exists()

        # Check frames were saved (in per-field subdirectories)
        frames = list((output_dir / "frames").glob("**/*.png"))
        # Each field has totalFrames, so multiply by number of fields
        num_fields = len(metadata["visualization"]["whatToPlot"])
        assert len(frames) == metadata["simulation"]["totalFrames"] * num_fields

    def test_run_with_different_solvers(self, tmp_path):
        """Test that different solvers can be used."""
        for solver in ["euler", "rk4"]:
            config_dict = {
                "preset": "heat",
                "parameters": {"D_T": 0.01},
                "init": {"type": "random-uniform", "params": {}},
                "solver": solver,
                "backend": "numpy",
                "t_end": 0.001,
                "dt": 0.0001,
                "resolution": [16, 16],
                "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
                "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
                "seed": 42,
            }

            config_path = tmp_path / f"config_{solver}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            config = load_config(config_path)
            runner = SimulationRunner(config, output_dir=tmp_path)

            # Should not raise
            metadata = runner.run(verbose=False)
            assert metadata["parameters"]["timesteppingScheme"] == solver.title()


class TestRunFromConfig:
    """Tests for run_from_config function."""

    def test_run_from_config_file(self, tmp_path):
        """Test running simulation from config file."""
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "backend": "numpy",
            "t_end": 0.002,  # 20 * 0.0001
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 6, "formats": ["png"]},
            "seed": 123,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        metadata = run_from_config(config_path, verbose=False)

        assert metadata is not None
        assert metadata["preset"] == "heat"

    def test_run_with_seed_override(self, tmp_path):
        """Test that seed can be overridden."""
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "backend": "numpy",
            "t_end": 0.001,  # 10 * 0.0001
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Override seed
        metadata = run_from_config(config_path, seed=999, verbose=False)
        assert metadata is not None


class TestGrayScottSimulation:
    """Tests for Gray-Scott simulation."""

    def test_gray_scott_short_simulation(self, tmp_path):
        """Test running a short Gray-Scott simulation."""
        config_dict = {
            "preset": "gray-scott",
            "parameters": {"a": 0.037, "b": 0.06, "D": 2.0},
            "init": {
                "type": "gaussian-blob",
                "params": {"num_blobs": 2, "positions": "random"},
            },
            "solver": "euler",
            "backend": "numpy",
            "t_end": 1.0,  # 100 * 0.01
            "dt": 0.01,  # Reduced for CFL stability (was 0.5)
            "resolution": [32, 32],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {
                "path": str(tmp_path),
                "num_frames": 11,
                "formats": ["png"],
            },
            "seed": 42,
            "domain_size": [2.5, 2.5],
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        metadata = runner.run(verbose=False)

        # Check metadata
        assert metadata["preset"] == "gray-scott"
        assert metadata["parameters"]["numSpecies"] == 2
        # With new multi-field output, default includes all fields when none specified
        assert metadata["visualization"]["whatToPlot"] == ["u", "v"]


class TestBackend:
    """Tests for backend configuration."""

    @pytest.mark.parametrize("backend", ["numpy", "numba", "auto"])
    def test_backend_runs(self, tmp_path, backend):
        """Test that different backends can be used."""
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "t_end": 0.0005,  # 10 * 0.0001
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
            "backend": backend,
        }

        config_path = tmp_path / f"config_{backend}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        # Should not raise
        metadata = runner.run(verbose=False)
        assert metadata["parameters"]["backend"] == backend

    def test_invalid_backend_raises(self, tmp_path):
        """Test that invalid backend raises ValueError."""
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "t_end": 0.0005,  # 5 * 0.0001
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
            "backend": "invalid_backend",
        }

        config_path = tmp_path / "config_invalid.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        with pytest.raises(ValueError, match="Invalid backend"):
            SimulationRunner(config, output_dir=tmp_path)


class TestAdaptiveTimeStepping:
    """Tests for adaptive time-stepping."""

    def test_adaptive_config_parsing(self, tmp_path):
        """Test that adaptive and tolerance are parsed from config."""
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "t_end": 0.0005,  # 5 * 0.0001
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
            "adaptive": True,
            "tolerance": 1e-5,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        assert config.adaptive is True
        assert config.tolerance == 1e-5

    def test_adaptive_simulation_runs(self, tmp_path):
        """Test that adaptive time stepping works."""
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "backend": "numpy",
            "t_end": 0.0005,  # 5 * 0.0001
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 6, "formats": ["png"]},
            "seed": 42,
            "adaptive": True,
            "tolerance": 1e-4,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        # Should not raise
        metadata = runner.run(verbose=False)
        assert metadata["parameters"]["adaptive"] is True
        assert metadata["parameters"]["tolerance"] == 1e-4


class TestUnusedParameterWarning:
    """Tests for unused parameter warning."""

    def test_warns_on_unused_parameter(self, tmp_path):
        """Test that a warning is raised for unused parameters."""
        config_dict = {
            "preset": "heat",
            "parameters": {
                "D_T": 0.01,  # Known parameter
                "unknown_param": 999,  # Unknown parameter
            },
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "backend": "numpy",
            "t_end": 0.0005,
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimulationRunner(config, output_dir=tmp_path)

            # Check that a warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "unknown_param" in str(w[0].message)
            assert "heat" in str(w[0].message)

    def test_warns_on_multiple_unused_parameters(self, tmp_path):
        """Test warning includes all unused parameters."""
        config_dict = {
            "preset": "heat",
            "parameters": {
                "D_T": 0.01,
                "foo": 1.0,
                "bar": 2.0,
            },
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "backend": "numpy",
            "t_end": 0.001,
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimulationRunner(config, output_dir=tmp_path)

            assert len(w) == 1
            msg = str(w[0].message)
            assert "foo" in msg
            assert "bar" in msg

    def test_no_warning_when_all_params_used(self, tmp_path):
        """Test that no warning is raised when all parameters are known."""
        config_dict = {
            "preset": "heat",
            "parameters": {"D_T": 0.01},  # Only known parameter
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "backend": "numpy",
            "t_end": 0.001,
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SimulationRunner(config, output_dir=tmp_path)

            # No warning should be raised
            assert len(w) == 0


class TestMissingParameterValidation:
    """Tests for missing parameter validation."""

    def test_missing_required_parameter(self, tmp_path):
        """Test that missing parameters raise a clear error."""
        config_dict = {
            "preset": "heat",
            "parameters": {},  # Missing required D_T parameter
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "backend": "numba",
            "adaptive": False,
            "t_end": 0.001,
            "dt": 0.0001,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)

        with pytest.raises(ValueError, match="Missing required parameters.*D_T"):
            SimulationRunner(config, output_dir=tmp_path)

    def test_missing_multiple_parameters(self, tmp_path):
        """Test that missing parameters error includes all missing params."""
        config_dict = {
            "preset": "gray-scott",
            "parameters": {"a": 0.037},  # Missing b and D parameters
            "init": {
                "type": "gaussian-blob",
                "params": {"num_blobs": 2, "positions": "random"},
            },
            "solver": "euler",
            "backend": "numpy",
            "adaptive": False,
            "t_end": 0.1,
            "dt": 0.01,
            "resolution": [16, 16],
            "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
            "output": {"path": str(tmp_path), "num_frames": 3, "formats": ["png"]},
            "seed": 42,
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)

        with pytest.raises(ValueError, match="Missing required parameters.*gray-scott"):
            SimulationRunner(config, output_dir=tmp_path)
