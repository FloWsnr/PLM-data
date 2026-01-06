"""Tests for 2D Navier-Stokes PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestNavierStokesPDE:
    """Tests for 2D Navier-Stokes equations."""

    def test_registered(self):
        """Test that navier-stokes is registered."""
        assert "navier-stokes" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("navier-stokes")
        meta = preset.metadata

        assert meta.name == "navier-stokes"
        assert meta.category == "fluids"
        assert meta.num_fields == 4
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "p" in meta.field_names
        assert "S" in meta.field_names

    def test_default_parameters(self):
        """Test default parameters match reference."""
        preset = get_pde_preset("navier-stokes")
        params = preset.get_default_parameters()

        assert "nu" in params
        assert "M" in params
        assert "D" in params
        assert params["nu"] == 0.02
        assert params["M"] == 0.5
        assert params["D"] == 0.05

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("navier-stokes")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state_shear_layer(self, small_grid):
        """Test shear layer initial condition."""
        preset = get_pde_preset("navier-stokes")
        state = preset.create_initial_state(
            small_grid, "shear-layer", {"amplitude": 0.5}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 4
        # u should have variation (shear layer)
        assert np.std(state[0].data) > 0
        # All fields should be finite
        for field in state:
            assert np.isfinite(field.data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("navier-stokes")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "shear-layer", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        # Check all fields are finite
        for field in result:
            assert np.isfinite(field.data).all()
