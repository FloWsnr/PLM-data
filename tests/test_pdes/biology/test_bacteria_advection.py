"""Tests for Bacteria Advection PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestBacteriaAdvectionPDE:
    """Tests for Bacteria Advection PDE."""

    def test_registered(self):
        """Test that bacteria-advection is registered."""
        assert "bacteria-advection" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("bacteria-advection")
        meta = preset.metadata

        assert meta.name == "bacteria-advection"
        assert meta.category == "biology"
        assert meta.num_fields == 1
        assert "C" in meta.field_names

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("bacteria-advection")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "uniform", {"c0": 0.5})

        result = pde.solve(state, t_range=0.1, dt=0.01, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
