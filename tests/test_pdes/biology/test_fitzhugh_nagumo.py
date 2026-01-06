"""Tests for FitzHugh-Nagumo PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestFitzHughNagumoPDE:
    """Tests for FitzHugh-Nagumo PDE."""

    def test_registered(self):
        """Test that fitzhugh-nagumo is registered."""
        assert "fitzhugh-nagumo" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fitzhugh-nagumo")
        meta = preset.metadata

        assert meta.name == "fitzhugh-nagumo"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("fitzhugh-nagumo")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
