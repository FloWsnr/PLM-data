"""Tests for vortex near wall (method of images) PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestMethodOfImagesPDE:
    """Tests for vortex near wall (method of images)."""

    def test_registered(self):
        """Test that method-of-images is registered."""
        assert "method-of-images" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("method-of-images")
        meta = preset.metadata

        assert meta.name == "method-of-images"
        assert meta.category == "fluids"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("method-of-images")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
