"""Tests for diffusively coupled Lorenz system."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets

from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestLorenzPDE:
    """Tests for diffusively coupled Lorenz system."""

    def test_registered(self):
        """Test that lorenz is registered."""
        assert "lorenz" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("lorenz")
        meta = preset.metadata

        assert meta.name == "lorenz"
        assert meta.category == "physics"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"X", "Y", "Z"}

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("lorenz")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.1})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self):
        """Test running a short simulation with the Lorenz PDE using default config.

        Field names have been changed from (x, y, z) to (X, Y, Z) to avoid
        collision with 2D grid coordinate names.
        """
        result, config = run_short_simulation("lorenz", "physics", t_end=0.001)

        assert isinstance(result, FieldCollection)
        assert len(result) == 3
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert np.isfinite(result[2].data).all()
        assert config["preset"] == "lorenz"
