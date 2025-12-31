"""Tests for Turing instability demo model."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestTuringConditionsPDE:
    """Tests for Turing instability demo model."""

    def test_registered(self):
        """Test that turing-conditions is registered."""
        assert "turing-conditions" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("turing-conditions")
        meta = preset.metadata

        assert meta.name == "turing-conditions"
        assert meta.category == "biology"
        assert meta.num_fields == 2

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("turing-conditions")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("turing-conditions")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use smaller timestep for stability
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
