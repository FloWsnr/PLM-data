"""Tests for inviscid Burgers equation PDE."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.inviscid_burgers import InviscidBurgersPDE

from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestInviscidBurgersPDE:
    """Tests for the inviscid Burgers equation preset."""

    def test_registered(self):
        """Test that inviscid-burgers is registered."""
        assert "inviscid-burgers" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("inviscid-burgers")
        meta = preset.metadata

        assert meta.name == "inviscid-burgers"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde_inviscid(self):
        """Test PDE creation with zero viscosity."""
        preset = get_pde_preset("inviscid-burgers")
        grid = create_grid_for_dimension(1, resolution=32)
        bc = create_bc_for_dimension(1)
        params = {"c": 1.0}

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None

    def test_create_pde_with_regularization(self):
        """Test PDE creation with small viscosity for regularization."""
        preset = get_pde_preset("inviscid-burgers")
        grid = create_grid_for_dimension(1, resolution=32)
        bc = create_bc_for_dimension(1)
        params = {"c": 1.0}

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None

    def test_create_initial_state_default(self):
        """Test default initial state (Gaussian pulse)."""
        preset = get_pde_preset("inviscid-burgers")
        grid = create_grid_for_dimension(1, resolution=32)

        state = preset.create_initial_state(grid, "default", {"seed": 42})

        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()
        # Should have small offset (0.001) plus Gaussian
        assert state.data.min() >= 0

    def test_create_initial_state_2d(self):
        """Test initial state creation in 2D."""
        preset = get_pde_preset("inviscid-burgers")
        grid = create_grid_for_dimension(2, resolution=16)

        state = preset.create_initial_state(grid, "default", {"seed": 42})

        assert isinstance(state, ScalarField)
        assert state.data.shape == (16, 16)
        assert np.isfinite(state.data).all()

    def test_short_simulation_with_regularization(self):
        """Test running a short simulation with small viscosity."""
        np.random.seed(42)
        preset = InviscidBurgersPDE()
        grid = create_grid_for_dimension(1, resolution=32)
        bc = create_bc_for_dimension(1)

        # Use default parameters
        params = {"c": 1.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "default", {"seed": 42})

        result = pde.solve(state, t_range=0.01, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test inviscid Burgers equation works in all supported dimensions."""
        np.random.seed(42)
        preset = InviscidBurgersPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Use default parameters
        params = {"c": 1.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.5})

        # Run short simulation with small timestep (hyperbolic PDE)
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "inviscid-burgers", ndim)
        check_dimension_variation(result, ndim, "inviscid-burgers")
