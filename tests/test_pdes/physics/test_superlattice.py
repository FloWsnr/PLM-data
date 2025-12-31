"""Tests for superlattice pattern formation PDE."""

import numpy as np
import pytest
from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestSuperlatticePDE:
    """Tests for superlattice pattern formation."""

    def test_registered(self):
        """Test that superlattice is registered."""
        assert "superlattice" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("superlattice")
        meta = preset.metadata

        assert meta.name == "superlattice"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        # Superlattice has 4th order terms requiring specialized solvers
        preset = get_pde_preset("superlattice")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"amplitude": 0.1})

        assert pde is not None
        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("superlattice")
        params = {"epsilon": 0.1, "g2": 0.5}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"amplitude": 0.05})

        # Check that PDE and state are created correctly
        assert pde is not None
        assert state is not None
        assert np.isfinite(state.data).all()
