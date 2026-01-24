"""Tests for bistable advection PDE."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.bistable_advection import BistableAdvectionPDE

from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestBistableAdvectionPDE:
    """Tests for the bistable advection equation preset."""

    def test_registered(self):
        """Test that bistable-advection is registered."""
        assert "bistable-advection" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("bistable-advection")
        meta = preset.metadata

        assert meta.name == "bistable-advection"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        preset = get_pde_preset("bistable-advection")
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)
        params = {"D": 0.1, "epsilon": 0.01, "vx": 1.0, "vy": 0.0}

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None

    def test_create_initial_state_default(self):
        """Test default initial state (step function)."""
        preset = get_pde_preset("bistable-advection")
        grid = create_grid_for_dimension(2, resolution=16)

        state = preset.create_initial_state(grid, "default", {"seed": 42})

        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()
        # Should have values near 0 and 1 (step function)
        assert state.data.min() >= -0.1
        assert state.data.max() <= 1.1

    def test_short_simulation(self):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = BistableAdvectionPDE()
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        pde = preset.create_pde({"D": 0.1, "epsilon": 0.01, "vx": 1.0, "vy": 0.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.0, "high": 1.0})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()

    def test_dimension_support_2d_only(self):
        """Test that bistable-advection only supports 2D (uses d_dy)."""
        preset = BistableAdvectionPDE()
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)

    def test_dimension_2d_simulation(self):
        """Test bistable-advection simulation in 2D."""
        np.random.seed(42)
        preset = BistableAdvectionPDE()

        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        pde = preset.create_pde({"D": 0.1, "epsilon": 0.01, "vx": 1.0, "vy": 0.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "bistable-advection", 2)
        check_dimension_variation(result, 2, "bistable-advection")
