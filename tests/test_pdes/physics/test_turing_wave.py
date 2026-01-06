"""Tests for hyperbolic Turing-wave pattern interaction PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestTuringWavePDE:
    """Tests for hyperbolic Turing-wave pattern interaction."""

    def test_registered(self):
        """Test that turing-wave is registered."""
        assert "turing-wave" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("turing-wave")
        meta = preset.metadata

        assert meta.name == "turing-wave"
        assert meta.category == "physics"
        assert meta.num_fields == 4  # Now hyperbolic: u, v, w, z
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names
        assert "z" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("turing-wave")
        params = preset.get_default_parameters()

        assert "tau" in params  # Hyperbolic coefficient
        assert "Du" in params
        assert "Dv" in params
        assert "a" in params
        assert "b" in params

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("turing-wave")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 4
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("turing-wave")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use very small timestep for stability (hyperbolic systems need careful numerics)
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert len(result) == 4
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
