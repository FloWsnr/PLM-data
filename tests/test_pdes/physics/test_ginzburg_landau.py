"""Tests for Complex Ginzburg-Landau PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestGinzburgLandauPDE:
    """Tests for the Complex Ginzburg-Landau equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("ginzburg-landau")
        meta = preset.metadata

        assert meta.name == "ginzburg-landau"
        assert meta.category == "physics"
        assert meta.num_fields == 2  # Now uses real/imaginary form (u, v)
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("ginzburg-landau")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_real_imaginary_initial_condition(self, small_grid):
        """Test real/imaginary field initial condition."""
        preset = get_pde_preset("ginzburg-landau")
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1}
        )

        assert state is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 2  # u and v fields
        assert np.isfinite(state[0].data).all()
        assert np.isfinite(state[1].data).all()

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "ginzburg-landau" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("ginzburg-landau")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.05}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
