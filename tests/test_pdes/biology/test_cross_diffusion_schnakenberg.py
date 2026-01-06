"""Tests for Cross-Diffusion Schnakenberg PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestCrossDiffusionSchnakenbergPDE:
    """Tests for Cross-Diffusion Schnakenberg PDE."""

    def test_registered(self):
        """Test that cross-diffusion-schnakenberg is registered."""
        assert "cross-diffusion-schnakenberg" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cross-diffusion-schnakenberg")
        meta = preset.metadata

        assert meta.name == "cross-diffusion-schnakenberg"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("cross-diffusion-schnakenberg")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"seed": 42})

        result = pde.solve(state, t_range=0.1, dt=0.01, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
