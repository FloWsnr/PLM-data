"""Tests for Allee effect model."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestHarshEnvironmentPDE:
    """Tests for Allee effect model."""

    def test_registered(self):
        """Test that harsh-environment is registered."""
        assert "harsh-environment" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("harsh-environment")
        meta = preset.metadata

        assert meta.name == "harsh-environment"
        assert meta.category == "biology"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("harsh-environment")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
