"""Tests for Stochastic Gray-Scott PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestStochasticGrayScottPDE:
    """Tests for Stochastic Gray-Scott PDE."""

    def test_registered(self):
        """Test that stochastic-gray-scott is registered."""
        assert "stochastic-gray-scott" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("stochastic-gray-scott")
        meta = preset.metadata

        assert meta.name == "stochastic-gray-scott"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test creating the PDE object (deterministic mode)."""
        preset = get_pde_preset("stochastic-gray-scott")
        params = preset.get_default_parameters()
        params["sigma"] = 0.0  # Deterministic
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("stochastic-gray-scott")
        state = preset.create_initial_state(small_grid, "default", {"seed": 42})

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()
        assert np.isfinite(state[1].data).all()
