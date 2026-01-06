"""Tests for Bistable Allen-Cahn PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestBistableAllenCahnPDE:
    """Tests for Bistable Allen-Cahn PDE."""

    def test_registered(self):
        """Test that bistable-allen-cahn is registered."""
        assert "bistable-allen-cahn" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("bistable-allen-cahn")
        meta = preset.metadata

        assert meta.name == "bistable-allen-cahn"
        assert meta.category == "biology"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("bistable-allen-cahn")
        params = preset.get_default_parameters()

        assert "D" in params
        assert "a" in params
        assert params["D"] == 1.0
        assert params["a"] == 0.5

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("bistable-allen-cahn")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("bistable-allen-cahn")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "random-uniform", {"min_val": 0.0, "max_val": 1.0, "seed": 42}
        )

        result = pde.solve(state, t_range=0.1, dt=0.01, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
