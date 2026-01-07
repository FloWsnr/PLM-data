"""Tests for Shallow Water PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestShallowWaterPDE:
    """Tests for the Shallow Water PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("shallow-water")
        meta = preset.metadata

        assert meta.name == "shallow-water"
        assert meta.category == "fluids"
        # Real shallow water has 3 fields: h (height), u (x-velocity), v (y-velocity)
        assert meta.num_fields == 3
        assert meta.field_names == ["h", "u", "v"]

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("shallow-water")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state_drop(self, small_grid):
        """Test water drop initial condition."""
        preset = get_pde_preset("shallow-water")
        state = preset.create_initial_state(
            small_grid, "drop", {"amplitude": 0.5}
        )

        # Should be a FieldCollection with 3 fields
        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        h, u, v = state
        # Height should have peak at center
        assert np.max(h.data) > 0
        # Velocities should start at zero
        assert np.allclose(u.data, 0)
        assert np.allclose(v.data, 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "shallow-water" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        # shallow-water needs shorter time due to stability
        result, config = run_short_simulation("shallow-water", "fluids", t_end=0.001)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
        assert config["preset"] == "shallow-water"
