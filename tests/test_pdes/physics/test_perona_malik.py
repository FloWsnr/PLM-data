"""Tests for Perona-Malik edge-preserving diffusion PDE."""

import numpy as np
import pytest

from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestPeronaMalikPDE:
    """Tests for Perona-Malik edge-preserving diffusion."""

    def test_registered(self):
        """Test that perona-malik is registered."""
        assert "perona-malik" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("perona-malik")
        meta = preset.metadata

        assert meta.name == "perona-malik"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("perona-malik")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()
