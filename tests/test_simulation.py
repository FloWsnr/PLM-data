"""Tests for simulation runner."""

import json
import numpy as np
import pytest
from pathlib import Path

from pde_sim.core.config import load_config, SimulationConfig
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
            "parameters": {"D": 0.01},
            "init": {"type": "random-uniform", "params": {"low": 0.0, "high": 1.0}},
            "solver": "euler",
            "timesteps": 50,
            "dt": 0.0001,
            "resolution": 16,
            "bc": {"x": "periodic", "y": "periodic"},
            "output": {"path": str(tmp_path), "frames_per_save": 10},
            "seed": 42,
        }

        import yaml
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

        # Check output files exist
        output_dir = tmp_path / metadata["id"]
        assert (output_dir / "metadata.json").exists()
        assert (output_dir / "frames").exists()

        # Check frames were saved
        frames = list((output_dir / "frames").glob("*.png"))
        assert len(frames) == metadata["simulation"]["totalFrames"]

    def test_run_with_different_solvers(self, tmp_path):
        """Test that different solvers can be used."""
        for solver in ["euler", "rk4"]:
            config_dict = {
                "preset": "heat",
                "parameters": {"D": 0.01},
                "init": {"type": "random-uniform", "params": {}},
                "solver": solver,
                "timesteps": 10,
                "dt": 0.0001,
                "resolution": 16,
                "bc": {"x": "periodic", "y": "periodic"},
                "output": {"path": str(tmp_path), "frames_per_save": 5},
                "seed": 42,
            }

            import yaml
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
            "parameters": {"D": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "timesteps": 20,
            "dt": 0.0001,
            "resolution": 16,
            "bc": {"x": "periodic", "y": "periodic"},
            "output": {"path": str(tmp_path), "frames_per_save": 10},
            "seed": 123,
        }

        import yaml
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
            "parameters": {"D": 0.01},
            "init": {"type": "random-uniform", "params": {}},
            "solver": "euler",
            "timesteps": 10,
            "dt": 0.0001,
            "resolution": 16,
            "bc": {"x": "periodic", "y": "periodic"},
            "output": {"path": str(tmp_path), "frames_per_save": 5},
            "seed": 42,
        }

        import yaml
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
            "parameters": {"F": 0.04, "k": 0.06, "Du": 0.16, "Dv": 0.08},
            "init": {"type": "gaussian-blobs", "params": {"num_blobs": 2}},
            "solver": "euler",
            "timesteps": 50,
            "dt": 0.5,
            "resolution": 32,
            "bc": {"x": "periodic", "y": "periodic"},
            "output": {
                "path": str(tmp_path),
                "frames_per_save": 10,
                "field_to_plot": "v",
            },
            "seed": 42,
            "domain_size": 2.5,
        }

        import yaml
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        metadata = runner.run(verbose=False)

        # Check metadata
        assert metadata["preset"] == "gray-scott"
        assert metadata["parameters"]["numSpecies"] == 2
        assert metadata["visualization"]["whatToPlot"] == "v"
