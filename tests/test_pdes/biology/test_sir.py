"""Tests for SIR epidemiological model."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestSIRModelPDE:
    """Tests for SIR epidemiological model."""

    def test_registered(self):
        """Test that sir is registered."""
        assert "sir" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("sir")
        meta = preset.metadata

        assert meta.name == "sir"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"s", "i", "r"}

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("sir")
        params = preset.get_default_parameters()

        assert "beta" in params
        assert "gamma" in params
        assert "D" in params
        assert params["beta"] == 2.0
        assert params["gamma"] == 0.1
        assert params["D"] == 0.1

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("sir")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_default(self, small_grid):
        """Test creating default initial state with localized infection."""
        preset = get_pde_preset("sir")
        state = preset.create_initial_state(
            small_grid, "default", {"location": "corner", "seed": 42}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert state[0].label == "s"
        assert state[1].label == "i"
        assert state[2].label == "r"
        # Most of population should be susceptible
        assert np.mean(state[0].data) > 0.9
        # Some infected in corner
        assert np.max(state[1].data) > 0

    def test_create_initial_state_random(self, small_grid):
        """Test creating random infection initial state."""
        preset = get_pde_preset("sir")
        state = preset.create_initial_state(
            small_grid, "random", {"infection_fraction": 0.1, "seed": 42}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        # Should have roughly 10% infected
        infected_fraction = np.mean(state[1].data > 0)
        assert infected_fraction > 0.05

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("sir")
        params = {"beta": 2.0, "gamma": 0.1, "D": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "default", {"location": "corner", "seed": 42}
        )

        result = pde.solve(state, t_range=0.1, dt=0.01, solver="euler")

        assert isinstance(result, FieldCollection)
        assert len(result) == 3
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert np.isfinite(result[2].data).all()
