"""Tests for Brusselator PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestBrusselatorPDE:
    """Tests for Brusselator PDE."""

    def test_registered(self):
        """Test that brusselator is registered."""
        assert "brusselator" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("brusselator")
        meta = preset.metadata

        assert meta.name == "brusselator"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("brusselator")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use smaller timestep for stability
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
