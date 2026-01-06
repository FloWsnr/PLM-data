"""Tests for superlattice pattern formation PDE (coupled Brusselator + Lengyel-Epstein)."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

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
        assert meta.num_fields == 4  # Coupled 4-field system
        assert "u1" in meta.field_names
        assert "v1" in meta.field_names
        assert "u2" in meta.field_names
        assert "v2" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("superlattice")
        params = preset.get_default_parameters()

        assert "a" in params
        assert "b" in params
        assert "c" in params
        assert "d" in params
        assert "alpha" in params
        # New parameter names with underscores
        assert "D_uone" in params
        assert "D_utwo" in params
        assert "D_uthree" in params
        assert "D_ufour" in params

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("superlattice")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.1})

        assert pde is not None
        assert state is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 4
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("superlattice")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert len(result) == 4
        assert np.isfinite(result[0].data).all()
