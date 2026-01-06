"""Tests for Cahn-Hilliard phase separation PDE."""

import numpy as np
import pytest
from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestCahnHilliardPDE:
    """Tests for Cahn-Hilliard phase separation."""

    def test_registered(self):
        """Test that cahn-hilliard is registered."""
        assert "cahn-hilliard" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cahn-hilliard")
        meta = preset.metadata

        assert meta.name == "cahn-hilliard"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("cahn-hilliard")
        params = preset.get_default_parameters()

        # New parameter names from reference
        assert "r" in params
        assert "a" in params
        assert "D" in params
        assert params["r"] == 0.01
        assert params["a"] == 1.0

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("cahn-hilliard")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("cahn-hilliard")
        state = preset.create_initial_state(
            small_grid, "random-uniform", {"low": -0.1, "high": 0.1}
        )

        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("cahn-hilliard")
        params = {"r": 0.01, "a": 1.0, "D": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "random-uniform", {"low": -0.05, "high": 0.05}
        )

        # Cahn-Hilliard has 4th order terms, needs small dt
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()
