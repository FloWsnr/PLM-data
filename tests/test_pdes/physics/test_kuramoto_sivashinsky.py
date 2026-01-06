"""Tests for Kuramoto-Sivashinsky PDE."""

import numpy as np
import pytest

from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestKuramotoSivashinskyPDE:
    """Tests for the Kuramoto-Sivashinsky equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("kuramoto-sivashinsky")
        meta = preset.metadata

        assert meta.name == "kuramoto-sivashinsky"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("kuramoto-sivashinsky")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "kuramoto-sivashinsky" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        # KS equation is fourth-order and very stiff - use coarse grid and tiny dt
        ks_grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)

        preset = get_pde_preset("kuramoto-sivashinsky")
        params = {"a": 0.1}  # Use damping for stability
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, ks_grid)
        state = preset.create_initial_state(
            ks_grid, "default", {"amplitude": 0.01, "seed": 42}
        )

        # Run a very short simulation with tiny timestep (fourth-order PDE is stiff)
        result = pde.solve(state, t_range=0.001, dt=1e-6)

        # Check that result is finite and valid
        assert result is not None
        assert np.isfinite(result.data).all(), "Simulation produced NaN or Inf values"
