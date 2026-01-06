"""Tests for Schnakenberg PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestSchnakenbergPDE:
    """Tests for Schnakenberg PDE."""

    def test_registered(self):
        """Test that schnakenberg is registered."""
        assert "schnakenberg" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("schnakenberg")
        meta = preset.metadata

        assert meta.name == "schnakenberg"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("schnakenberg")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use very small timestep for stability
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
