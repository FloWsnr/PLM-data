"""Tests for advection-diffusion equation with rotational flow PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestAdvectionRotationalPDE:
    """Tests for the rotational (vortex) advection-diffusion equation preset."""

    def test_registered(self):
        """Test that advection-rotational PDE is registered."""
        assert "advection-rotational" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("advection-rotational")
        meta = preset.metadata

        assert meta.name == "advection-rotational"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation with angular velocity."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)
        preset = get_pde_preset("advection-rotational")
        pde = preset.create_pde(
            parameters={"D": 1.0, "omega": 0.5},
            bc={"x": "dirichlet", "y": "dirichlet"},
            grid=grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test advection-rotational equation works in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("advection-rotational")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Use non-periodic for rotational flow
        grid = create_grid_for_dimension(ndim, resolution=16, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        pde = preset.create_pde({"D": 0.0, "omega": 1.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "advection-rotational", ndim)
        check_dimension_variation(result, ndim, "advection-rotational")

    def test_unsupported_dimensions(self):
        """Test that advection-rotational rejects 1D and 3D."""
        preset = get_pde_preset("advection-rotational")
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
