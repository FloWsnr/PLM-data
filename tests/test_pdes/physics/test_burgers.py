"""Tests for Burgers' equation PDE."""

import numpy as np
import pytest
from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestBurgersPDE:
    """Tests for Burgers' equation."""

    def test_registered(self):
        """Test that burgers is registered."""
        assert "burgers" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("burgers")
        meta = preset.metadata

        assert meta.name == "burgers"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("burgers")
        params = preset.get_default_parameters()

        assert "nu" in params
        assert params["nu"] == 0.01

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("burgers")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("burgers")
        state = preset.create_initial_state(
            small_grid, "sine", {"wavelength": 0.5}
        )

        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("burgers")
        params = {"nu": 0.1}  # Higher viscosity for stability
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "sine", {"wavelength": 0.5}
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()
