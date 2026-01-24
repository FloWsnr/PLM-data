"""Tests for 2D Navier-Stokes PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


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

    def test_create_initial_state_poiseuille(self, small_grid):
        """Test Poiseuille (parabolic) initial condition."""
        preset = get_pde_preset("navier-stokes")
        state = preset.create_initial_state(
            small_grid, "poiseuille", {"amplitude": 0.4}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 4
        u, v, p, S = state

        # u should have variation (parabolic profile)
        assert np.std(u.data) > 0
        # u should be negative (flow in -x direction)
        assert np.min(u.data) < 0

        # v should be zero (no vertical velocity)
        assert np.allclose(v.data, 0)

        # p should have variation (pressure gradient)
        assert np.std(p.data) > 0

        # All fields should be finite
        for field in state:
            assert np.isfinite(field.data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("navier-stokes", "fluids")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
        assert config["preset"] == "navier-stokes"

    def test_dimension_support_2d_only(self):
        """Test that navier-stokes only supports 2D."""
        preset = get_pde_preset("navier-stokes")

        # Verify only 2D is supported
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
