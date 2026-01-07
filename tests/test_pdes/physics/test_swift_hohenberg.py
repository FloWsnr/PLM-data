"""Tests for Swift-Hohenberg pattern formation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets

from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestSwiftHohenbergPDE:
    """Tests for Swift-Hohenberg pattern formation."""

    def test_registered(self):
        """Test that swift-hohenberg is registered."""
        assert "swift-hohenberg" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("swift-hohenberg")
        meta = preset.metadata

        assert meta.name == "swift-hohenberg"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("swift-hohenberg")
        params = preset.get_default_parameters()

        assert "r" in params
        assert "a" in params
        assert "b" in params
        assert "c" in params
        assert "D" in params

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("swift-hohenberg")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("swift-hohenberg")
        state = preset.create_initial_state(
            small_grid, "random-uniform", {"low": -0.1, "high": 0.1}
        )

        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config.

        Swift-Hohenberg has 4th order terms making it numerically stiff.
        """
        # Run a very short simulation (fourth-order PDE is stiff)
        result, config = run_short_simulation("swift-hohenberg", "physics", t_end=0.001)

        # Check that result is finite and valid
        assert result is not None
        assert np.isfinite(result.data).all(), "Simulation produced NaN or Inf values"
        assert config["preset"] == "swift-hohenberg"

    def test_subcritical_with_quintic(self, small_grid):
        """Test subcritical regime with quintic term.

        Parameters for subcritical localised patterns: r<0, a>0, b<0, c<0.
        """
        # Use larger domain for stability
        sh_grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)

        preset = get_pde_preset("swift-hohenberg")
        params = {"r": -0.1, "a": 0.5, "b": -1.0, "c": -0.1, "D": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, sh_grid)
        state = preset.create_initial_state(
            sh_grid, "random-uniform", {"low": -0.05, "high": 0.05, "seed": 42}
        )

        # Run a very short simulation with tiny timestep (fourth-order PDE is stiff)
        result = pde.solve(state, t_range=0.001, dt=1e-6)

        # Check that result is finite and valid
        assert result is not None
        assert np.isfinite(result.data).all(), "Simulation produced NaN or Inf values"
