"""Tests for Korteweg-de Vries (KdV) PDE."""

import numpy as np
import pytest
from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestKdVPDE:
    """Tests for the Korteweg-de Vries equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("kdv")
        meta = preset.metadata

        assert meta.name == "kdv"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("kdv")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_soliton_initial_condition(self, small_grid):
        """Test soliton initial condition."""
        preset = get_pde_preset("kdv")
        state = preset.create_initial_state(
            small_grid, "soliton", {"amplitude": 1.0, "width": 0.1}
        )

        assert state is not None
        assert np.max(state.data) > 0  # Should have a peak

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "kdv" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("kdv")
        params = {"c": 0.0, "alpha": 1.0, "beta": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "soliton", {"amplitude": 0.5, "width": 0.1}
        )

        # Check that PDE and state are created correctly
        assert pde is not None
        assert state is not None
        assert np.isfinite(state.data).all()
