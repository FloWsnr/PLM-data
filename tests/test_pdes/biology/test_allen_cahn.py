"""Tests for Allen-Cahn PDEs."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestAllenCahnPDE:
    """Tests for Allen-Cahn PDE."""

    def test_registered(self):
        """Test that allen-cahn is registered."""
        assert "allen-cahn" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("allen-cahn")
        meta = preset.metadata

        assert meta.name == "allen-cahn"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("allen-cahn")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "random-uniform", {"min_val": -1.0, "max_val": 1.0})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestStandardAllenCahnPDE:
    """Tests for standard Allen-Cahn with symmetric double-well."""

    def test_registered(self):
        """Test that allen-cahn-standard is registered."""
        assert "allen-cahn-standard" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("allen-cahn-standard")
        meta = preset.metadata

        assert meta.name == "allen-cahn-standard"
        assert meta.category == "biology"
        assert meta.num_fields == 1
        assert "c" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("allen-cahn-standard")
        params = preset.get_default_parameters()

        assert "gamma" in params
        assert "mobility" in params
        assert params["gamma"] == 1.0
        assert params["mobility"] == 1.0

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("allen-cahn-standard")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_default(self, small_grid):
        """Test creating default initial state (random noise)."""
        preset = get_pde_preset("allen-cahn-standard")
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1, "seed": 42}
        )

        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()
        # Should be small amplitude noise around 0
        assert np.abs(np.mean(state.data)) < 0.5

    def test_create_initial_state_step(self, small_grid):
        """Test creating step function initial state."""
        preset = get_pde_preset("allen-cahn-standard")
        state = preset.create_initial_state(small_grid, "step", {})

        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()
        # Should have -1 on left, +1 on right
        assert np.min(state.data) == -1.0
        assert np.max(state.data) == 1.0

    def test_create_initial_state_tanh(self, small_grid):
        """Test creating tanh profile initial state."""
        preset = get_pde_preset("allen-cahn-standard")
        state = preset.create_initial_state(small_grid, "tanh", {"gamma": 1.0})

        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()
        # Should have smooth transition with negative values on left, positive on right
        # Note: on a [0,1] domain the tanh doesn't reach ±1, but should show transition
        assert np.min(state.data) < 0
        assert np.max(state.data) > 0
        # Mean should be close to 0 (symmetric around center)
        assert np.abs(np.mean(state.data)) < 0.1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("allen-cahn-standard")
        params = {"gamma": 1.0, "mobility": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1, "seed": 42}
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
