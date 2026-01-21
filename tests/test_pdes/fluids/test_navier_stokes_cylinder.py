"""Tests for Navier-Stokes flow around cylinder PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestNavierStokesCylinderPDE:
    """Tests for Navier-Stokes cylinder flow equations."""

    def test_registered(self):
        """Test that navier-stokes-cylinder is registered."""
        assert "navier-stokes-cylinder" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("navier-stokes-cylinder")
        meta = preset.metadata

        assert meta.name == "navier-stokes-cylinder"
        assert meta.category == "fluids"
        assert meta.num_fields == 4
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "p" in meta.field_names
        assert "S" in meta.field_names

    def test_default_parameters(self):
        """Test default parameters match reference."""
        preset = get_pde_preset("navier-stokes-cylinder")
        params = preset.get_default_parameters()

        assert "nu" in params
        assert "M" in params
        assert "U" in params
        assert "cylinder_radius" in params
        assert params["nu"] == 0.1
        assert params["M"] == 0.5
        assert params["U"] == 0.7
        assert params["cylinder_radius"] == 0.05

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("navier-stokes-cylinder")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state_cylinder_flow(self, small_grid):
        """Test cylinder flow initial condition."""
        preset = get_pde_preset("navier-stokes-cylinder")
        state = preset.create_initial_state(
            small_grid,
            "cylinder-flow",
            {"U": 0.7, "cylinder_radius": 0.1, "cylinder_x": 0.5, "cylinder_y": 0.5},
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 4
        u, v, p, S = state

        # u should be positive (flow in +x direction) outside cylinder
        # Inside cylinder, u should be near zero (due to damping)
        assert np.max(u.data) > 0

        # S should be high inside cylinder, low outside
        # (with smooth tanh transition, max may not reach 1.0 on coarse grids)
        assert np.max(S.data) > 0.5  # Should have elevated value inside cylinder
        assert np.min(S.data) < 0.1  # Should have low value outside

        # All fields should be finite
        for field in state:
            assert np.isfinite(field.data).all()

    def test_cylinder_creates_obstacle(self, small_grid):
        """Test that cylinder creates proper obstacle region."""
        preset = get_pde_preset("navier-stokes-cylinder")
        state = preset.create_initial_state(
            small_grid,
            "cylinder-flow",
            {"U": 0.7, "cylinder_radius": 0.15, "cylinder_x": 0.5, "cylinder_y": 0.5},
        )

        u, v, p, S = state

        # Center of domain should have high S (inside cylinder)
        center_idx = small_grid.shape[0] // 2
        assert S.data[center_idx, center_idx] > 0.5

        # Corners should have low S (outside cylinder)
        assert S.data[0, 0] < 0.5
        assert S.data[-1, -1] < 0.5

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation(
            "navier-stokes-cylinder", "fluids", t_end=0.01
        )

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
        assert config["preset"] == "navier-stokes-cylinder"

    def test_dimension_support_2d_only(self):
        """Test that navier-stokes-cylinder only supports 2D."""
        preset = get_pde_preset("navier-stokes-cylinder")

        # Verify only 2D is supported
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
