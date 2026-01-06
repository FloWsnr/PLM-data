"""Tests for Nonlinear Schrodinger PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestNonlinearSchrodingerPDE:
    """Tests for Nonlinear Schrodinger PDE."""

    def test_registered(self):
        """Test that nonlinear-schrodinger is registered."""
        assert "nonlinear-schrodinger" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("nonlinear-schrodinger")
        meta = preset.metadata

        assert meta.name == "nonlinear-schrodinger"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("nonlinear-schrodinger")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"seed": 42})

        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
