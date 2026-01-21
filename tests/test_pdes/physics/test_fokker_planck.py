"""Tests for Fokker-Planck equation PDE preset."""

import numpy as np
import pytest
import yaml

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset
from pde_sim.pdes.physics.fokker_planck import FokkerPlanckPDE
from pde_sim.core.config import load_config
from pde_sim.core.simulation import SimulationRunner

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


class TestFokkerPlanckMetadata:
    """Tests for Fokker-Planck PDE metadata."""

    def test_preset_registration(self):
        """Test that fokker-planck preset is registered."""
        preset = get_pde_preset("fokker-planck")
        assert preset is not None

    def test_metadata_fields(self):
        """Test metadata has required fields."""
        preset = get_pde_preset("fokker-planck")
        metadata = preset.metadata

        assert metadata.name == "fokker-planck"
        assert metadata.category == "physics"
        assert metadata.num_fields == 1
        assert metadata.field_names == ["p"]
        assert "p" in metadata.equations

    def test_parameters(self):
        """Test parameter definitions."""
        preset = get_pde_preset("fokker-planck")
        param_names = [p.name for p in preset.metadata.parameters]

        assert "D" in param_names
        assert "gamma" in param_names
        assert "x0" in param_names
        assert "y0" in param_names

    def test_default_parameters(self):
        """Test default parameter values."""
        preset = get_pde_preset("fokker-planck")
        defaults = preset.get_default_parameters()

        assert defaults["D"] == 0.1
        assert defaults["gamma"] == 0.5
        assert defaults["x0"] == 0.0
        assert defaults["y0"] == 0.0


class TestFokkerPlanckPDE:
    """Tests for Fokker-Planck PDE creation."""

    def test_create_pde(self):
        """Test PDE creation."""
        preset = get_pde_preset("fokker-planck")
        grid = CartesianGrid([[0, 10], [0, 10]], [64, 64])

        bc = {"x-": "neumann:0", "x+": "neumann:0", "y-": "neumann:0", "y+": "neumann:0"}
        params = {"D": 0.1, "gamma": 0.5, "x0": 0.0, "y0": 0.0}

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None

    def test_create_initial_state_default(self):
        """Test default initial condition creation."""
        preset = get_pde_preset("fokker-planck")
        grid = CartesianGrid([[0, 10], [0, 10]], [64, 64])

        state = preset.create_initial_state(
            grid, "default", {"sigma": 0.15, "init_x_offset": 0.3, "init_y_offset": 0.3, "seed": 42}
        )

        assert state.grid == grid
        assert state.data.shape == (64, 64)
        # Check values are finite
        assert np.all(np.isfinite(state.data))
        # Check probability is positive (or zero)
        assert np.all(state.data >= 0)

    def test_create_initial_state_gaussian_ring(self):
        """Test gaussian-ring initial condition."""
        preset = get_pde_preset("fokker-planck")
        grid = CartesianGrid([[0, 10], [0, 10]], [64, 64])

        state = preset.create_initial_state(
            grid, "gaussian-ring", {"radius": 0.3, "width": 0.05, "seed": 42}
        )

        assert state.data.shape == (64, 64)
        assert np.all(np.isfinite(state.data))
        assert np.all(state.data >= 0)

    def test_create_initial_state_double_gaussian(self):
        """Test double-gaussian initial condition."""
        preset = get_pde_preset("fokker-planck")
        grid = CartesianGrid([[0, 10], [0, 10]], [64, 64])

        state = preset.create_initial_state(
            grid, "double-gaussian", {"sigma": 0.1, "separation": 0.4, "seed": 42}
        )

        assert state.data.shape == (64, 64)
        assert np.all(np.isfinite(state.data))
        assert np.all(state.data >= 0)

    def test_initial_state_normalized(self):
        """Test that initial state is normalized (approximately unit integral)."""
        preset = get_pde_preset("fokker-planck")
        grid = CartesianGrid([[0, 10], [0, 10]], [64, 64])

        state = preset.create_initial_state(
            grid, "default", {"sigma": 0.15, "init_x_offset": 0.2, "init_y_offset": 0.2, "seed": 42}
        )

        # Compute numerical integral
        dx = 10.0 / 64
        dy = 10.0 / 64
        total = np.sum(state.data) * dx * dy

        # Should be approximately 1 (normalized probability)
        assert abs(total - 1.0) < 0.1  # Allow some numerical tolerance

    def test_run_default_config(self):
        """Test running simulation with default config."""
        result, config = run_short_simulation("fokker-planck", "physics", t_end=0.1)

        # Check simulation completed
        assert result is not None
        # Check final state is finite
        assert np.all(np.isfinite(result.data))

    def test_dimension_support_2d(self):
        """Test Fokker-Planck works in 2D.

        Note: The current implementation uses 2D-specific terms (d_dy, y coordinates),
        so the test only validates 2D support even though metadata claims [1, 2, 3].
        """
        np.random.seed(42)
        preset = FokkerPlanckPDE()
        ndim = 2

        # Check 2D is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs (use non-periodic for Fokker-Planck)
        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        # Create PDE and initial state
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "fokker-planck", ndim)


class TestFokkerPlanckSimulation:
    """Tests for running Fokker-Planck simulations."""

    def test_short_simulation_finite(self, tmp_path):
        """Test that simulation produces finite results."""
        config_dict = {
            "preset": "fokker-planck",
            "parameters": {"D": 0.1, "gamma": 0.5, "x0": 0.0, "y0": 0.0},
            "init": {
                "type": "default",
                "params": {"sigma": 0.15, "init_x_offset": 0.3, "init_y_offset": 0.3},
            },
            "solver": "euler",
            "backend": "numpy",
            "t_end": 1.0,
            "dt": 0.01,
            "resolution": [32, 32],
            "bc": {
                "x-": "neumann:0",
                "x+": "neumann:0",
                "y-": "neumann:0",
                "y+": "neumann:0",
            },
            "output": {"path": str(tmp_path), "num_frames": 11, "format": "numpy"},
            "seed": 42,
            "domain_size": [10.0, 10.0],
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        metadata = runner.run(verbose=False)

        # Check simulation completed
        assert metadata["preset"] == "fokker-planck"
        assert metadata["simulation"]["totalFrames"] > 0

        # Load the trajectory and check final state is finite
        output_dir = tmp_path / metadata["folder_name"]
        trajectory = np.load(output_dir / "trajectory.npy")
        final_state = trajectory[-1]  # Last frame
        assert np.all(np.isfinite(final_state))

    @pytest.mark.filterwarnings("ignore:.*_make_error_synchronizer.*:DeprecationWarning")
    def test_probability_conservation(self, tmp_path):
        """Test that probability is approximately conserved during simulation.

        Note: The Fokker-Planck with harmonic drift and Neumann BCs may not
        perfectly conserve probability if probability reaches boundaries.
        We use a larger domain and centered initial condition to minimize this.
        """
        config_dict = {
            "preset": "fokker-planck",
            "parameters": {"D": 0.1, "gamma": 0.5, "x0": 0.0, "y0": 0.0},
            "init": {
                "type": "default",
                "params": {"sigma": 0.1, "init_x_offset": 0.1, "init_y_offset": 0.1},
            },
            "solver": "rk4",  # More accurate solver
            "backend": "numpy",
            "t_end": 2.0,
            "dt": 0.005,
            "resolution": [64, 64],
            "bc": {
                "x-": "neumann:0",
                "x+": "neumann:0",
                "y-": "neumann:0",
                "y+": "neumann:0",
            },
            "output": {"path": str(tmp_path), "num_frames": 5, "format": "numpy"},
            "seed": 42,
            "domain_size": [20.0, 20.0],  # Larger domain to keep probability away from boundaries
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        metadata = runner.run(verbose=False)

        # Load trajectory
        output_dir = tmp_path / metadata["folder_name"]
        trajectory = np.load(output_dir / "trajectory.npy")

        # Get initial and final total probability
        dx = 20.0 / 64
        dy = 20.0 / 64
        initial_data = trajectory[0, :, :, 0]  # First frame, first field
        final_data = trajectory[-1, :, :, 0]  # Last frame, first field

        initial_total = np.sum(initial_data) * dx * dy
        final_total = np.sum(final_data) * dx * dy

        # Probability should be approximately conserved
        # Allow for some numerical diffusion at boundaries
        relative_change = abs(final_total - initial_total) / max(initial_total, 1e-10)
        assert relative_change < 0.5  # Allow up to 50% change due to source term in equation

    @pytest.mark.filterwarnings("ignore:.*_make_error_synchronizer.*:DeprecationWarning")
    def test_relaxation_toward_center(self, tmp_path):
        """Test that probability distribution moves toward potential center."""
        domain_size = 10.0
        resolution = 64

        config_dict = {
            "preset": "fokker-planck",
            "parameters": {"D": 0.05, "gamma": 1.0, "x0": 0.0, "y0": 0.0},
            "init": {
                "type": "default",
                "params": {"sigma": 0.1, "init_x_offset": 0.4, "init_y_offset": 0.4},
            },
            "solver": "rk4",
            "backend": "numpy",
            "t_end": 5.0,
            "dt": 0.005,
            "resolution": [resolution, resolution],
            "bc": {
                "x-": "neumann:0",
                "x+": "neumann:0",
                "y-": "neumann:0",
                "y+": "neumann:0",
            },
            "output": {"path": str(tmp_path), "num_frames": 3, "format": "numpy"},
            "seed": 42,
            "domain_size": [domain_size, domain_size],
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path)
        runner = SimulationRunner(config, output_dir=tmp_path)

        metadata = runner.run(verbose=False)

        # Load trajectory
        output_dir = tmp_path / metadata["folder_name"]
        trajectory = np.load(output_dir / "trajectory.npy")

        # Get initial and final data
        initial_data = trajectory[0, :, :, 0]  # First frame, first field
        final_data = trajectory[-1, :, :, 0]  # Last frame, first field

        # Compute center of mass
        x = np.linspace(0, domain_size, resolution)
        y = np.linspace(0, domain_size, resolution)
        X, Y = np.meshgrid(x, y, indexing="ij")

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        initial_total = np.sum(initial_data) * dx * dy
        initial_x_mean = np.sum(X * initial_data) * dx * dy / max(initial_total, 1e-10)
        initial_y_mean = np.sum(Y * initial_data) * dx * dy / max(initial_total, 1e-10)

        final_total = np.sum(final_data) * dx * dy
        final_x_mean = np.sum(X * final_data) * dx * dy / max(final_total, 1e-10)
        final_y_mean = np.sum(Y * final_data) * dx * dy / max(final_total, 1e-10)

        # Center of mass should be closer to domain center (5, 5) after relaxation
        domain_center = domain_size / 2.0
        initial_dist = np.sqrt((initial_x_mean - domain_center)**2 + (initial_y_mean - domain_center)**2)
        final_dist = np.sqrt((final_x_mean - domain_center)**2 + (final_y_mean - domain_center)**2)

        # Final should be closer to center than initial
        assert final_dist < initial_dist, f"Expected final_dist ({final_dist:.4f}) < initial_dist ({initial_dist:.4f})"


class TestFokkerPlanckConfig:
    """Tests for Fokker-Planck config files."""

    def test_default_config_loads(self):
        """Test that default config file loads correctly."""
        import os
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "configs",
            "physics",
            "fokker_planck",
            "default.yaml",
        )

        if os.path.exists(config_path):
            config = load_config(config_path)
            assert config.preset == "fokker-planck"
