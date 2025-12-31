"""Tests for Shallow Water PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestShallowWaterPDE:
    """Tests for the Shallow Water PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("shallow-water")
        meta = preset.metadata

        assert meta.name == "shallow-water"
        assert meta.category == "fluids"
        assert meta.num_fields == 1
        assert meta.field_names == ["h"]

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("shallow-water")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state_drop(self, small_grid):
        """Test water drop initial condition."""
        preset = get_pde_preset("shallow-water")
        state = preset.create_initial_state(
            small_grid, "drop", {"amplitude": 0.5, "background": 1.0}
        )

        assert isinstance(state, ScalarField)
        # Should have values around background with peak at center
        assert np.max(state.data) > 1.0
        assert np.min(state.data) >= 1.0 - 0.01  # Near background

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "shallow-water" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("shallow-water")
        params = {"c": 0.5}  # Slower wave speed for stability
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "drop", {"amplitude": 0.1})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()
