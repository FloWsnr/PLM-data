"""Tests for Keller-Segel PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestKellerSegelPDE:
    """Tests for the Keller-Segel chemotaxis model."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("keller-segel")
        meta = preset.metadata

        assert meta.name == "keller-segel"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names  # cells
        assert "c" in meta.field_names  # chemoattractant

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("keller-segel")
        params = preset.get_default_parameters()

        assert "Du" in params
        assert "Dc" in params
        assert "chi" in params  # chemotactic sensitivity
        assert "alpha" in params  # production
        assert "beta" in params  # decay

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("keller-segel")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_default(self, small_grid):
        """Test default initial state creation."""
        preset = get_pde_preset("keller-segel")
        state = preset.create_initial_state(
            small_grid, "keller-segel-default", {}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "c"

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "keller-segel" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("keller-segel")
        # Use moderate parameters to avoid blow-up
        params = {"Du": 1.0, "Dc": 1.0, "chi": 0.5, "alpha": 0.5, "beta": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "keller-segel-default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
