"""Tests for Kardar-Parisi-Zhang interface growth equation."""

import numpy as np
import pytest
from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestKPZInterfacePDE:
    """Tests for Kardar-Parisi-Zhang interface growth equation."""

    def test_registered(self):
        """Test that kpz is registered."""
        assert "kpz" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("kpz")
        meta = preset.metadata

        assert meta.name == "kpz"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "h" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("kpz")
        params = preset.get_default_parameters()

        assert "nu" in params
        assert "lmbda" in params
        assert params["nu"] == 0.5
        assert params["lmbda"] == 1.0

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("kpz")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_default(self, small_grid):
        """Test creating default initial state."""
        preset = get_pde_preset("kpz")
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1, "seed": 42}
        )

        assert state is not None
        assert np.isfinite(state.data).all()

    def test_create_initial_state_sinusoidal(self, small_grid):
        """Test creating sinusoidal initial state."""
        preset = get_pde_preset("kpz")
        state = preset.create_initial_state(
            small_grid, "sinusoidal", {"amplitude": 1.0, "wavelength": 0.5}
        )

        assert state is not None
        assert np.isfinite(state.data).all()
        # Should have sinusoidal pattern with max/min values
        assert np.max(state.data) > 0.5
        assert np.min(state.data) < -0.5

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("kpz")
        params = {"nu": 0.5, "lmbda": 0.5}  # Moderate parameters for stability
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1, "seed": 42}
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()
