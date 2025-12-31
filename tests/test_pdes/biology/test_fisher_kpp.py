"""Tests for Fisher-KPP PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestFisherKPPPDE:
    """Tests for Fisher-KPP PDE."""

    def test_registered(self):
        """Test that fisher-kpp is registered."""
        assert "fisher-kpp" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fisher-kpp")
        meta = preset.metadata

        assert meta.name == "fisher-kpp"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("fisher-kpp")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "random-uniform", {"min_val": 0.0, "max_val": 1.0})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
