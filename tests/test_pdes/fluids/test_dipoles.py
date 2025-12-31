"""Tests for vortex dipole dynamics PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestDipolesPDE:
    """Tests for vortex dipole dynamics."""

    def test_registered(self):
        """Test that dipoles is registered."""
        assert "dipoles" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("dipoles")
        meta = preset.metadata

        assert meta.name == "dipoles"
        assert meta.category == "fluids"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("dipoles")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        # Dipole should have positive and negative regions
        assert np.any(result.data > 0)
        assert np.any(result.data < 0)
