"""Tests for Korteweg-de Vries PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestKortewegDeVriesPDE:
    """Tests for Korteweg-de Vries PDE."""

    def test_registered(self):
        """Test that korteweg-de-vries is registered."""
        assert "korteweg-de-vries" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("korteweg-de-vries")
        meta = preset.metadata

        assert meta.name == "korteweg-de-vries"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "phi" in meta.field_names

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("korteweg-de-vries")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"seed": 42})

        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
