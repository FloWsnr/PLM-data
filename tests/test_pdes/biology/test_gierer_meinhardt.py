"""Tests for Gierer-Meinhardt PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestGiererMeinhardtPDE:
    """Tests for the Gierer-Meinhardt activator-inhibitor system."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("gierer-meinhardt")
        meta = preset.metadata

        assert meta.name == "gierer-meinhardt"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("gierer-meinhardt")
        params = preset.get_default_parameters()

        assert "Du" in params
        assert "Dv" in params
        assert "a" in params  # basal production
        assert "b" in params  # decay rate
        assert "c" in params  # inhibitor decay
        assert params["Dv"] > params["Du"]  # Inhibitor should diffuse faster

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("gierer-meinhardt")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test initial state creation."""
        preset = get_pde_preset("gierer-meinhardt")
        state = preset.create_initial_state(
            small_grid, "default", {"noise": 0.01}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"
        # All values should be positive
        assert np.all(state[0].data > 0)
        assert np.all(state[1].data > 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "gierer-meinhardt" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("gierer-meinhardt")
        params = {"Du": 1.0, "Dv": 40.0, "a": 0.1, "b": 1.0, "c": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.01})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
