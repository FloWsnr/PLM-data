"""Tests for vorticity bounded PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.fluids.vorticity_bounded import VorticityBoundedPDE

from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


class TestVorticityBoundedPDE:
    """Tests for the vorticity bounded equation preset."""

    def test_registered(self):
        """Test that vorticity-bounded is registered."""
        assert "vorticity-bounded" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("vorticity-bounded")
        meta = preset.metadata

        assert meta.name == "vorticity-bounded"
        assert meta.category == "fluids"
        assert meta.num_fields == 3
        assert "omega" in meta.field_names
        assert "psi" in meta.field_names
        assert "S" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("vorticity-bounded")
        params = preset.get_default_parameters()

        assert "nu" in params
        assert "epsilon" in params
        assert "D" in params
        assert "k" in params
        assert params["nu"] == 0.1
        assert params["epsilon"] == 0.05
        assert params["D"] == 0.05
        assert params["k"] == 51

    def test_create_pde(self):
        """Test PDE creation."""
        preset = get_pde_preset("vorticity-bounded")
        grid = create_grid_for_dimension(2, resolution=16)
        bc = preset.get_default_bc()
        params = preset.get_default_parameters()

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None

    def test_create_initial_state(self):
        """Test initial state creation (oscillatory vorticity)."""
        preset = get_pde_preset("vorticity-bounded")
        grid = create_grid_for_dimension(2, resolution=32)

        state = preset.create_initial_state(grid, "default", {"k": 10})

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert state[0].label == "omega"
        assert state[1].label == "psi"
        assert state[2].label == "S"

    def test_get_default_bc(self):
        """Test that default boundary conditions are returned."""
        preset = VorticityBoundedPDE()
        bc = preset.get_default_bc()

        assert bc is not None
        # Check field-specific BCs exist
        assert bc.fields is not None
        assert "omega" in bc.fields
        assert "S" in bc.fields

    def test_short_simulation(self):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = VorticityBoundedPDE()
        grid = create_grid_for_dimension(2, resolution=16)
        bc = preset.get_default_bc()

        # Use smaller k for test stability
        params = preset.get_default_parameters()
        params["k"] = 4

        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "default", {"k": 4})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None)

        assert isinstance(result, FieldCollection)
        assert len(result) == 3
        for field in result:
            assert np.isfinite(field.data).all()

    def test_dimension_support_2d_only(self):
        """Test that vorticity-bounded only supports 2D."""
        preset = VorticityBoundedPDE()
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)

    def test_dimension_2d_simulation(self):
        """Test vorticity-bounded simulation in 2D."""
        np.random.seed(42)
        preset = VorticityBoundedPDE()

        grid = create_grid_for_dimension(2, resolution=16)
        bc = preset.get_default_bc()

        params = preset.get_default_parameters()
        params["k"] = 4  # Use smaller k for stability

        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "default", {"k": 4})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None)

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "vorticity-bounded", 2)
