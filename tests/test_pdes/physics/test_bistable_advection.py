"""Tests for bistable advection PDE."""

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
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("bistable-advection")
        pde = preset.create_pde(
            {"D": 0.1, "epsilon": 0.01, "vx": 1.0, "vy": 0.0},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test bistable-advection simulation in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("bistable-advection")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"D": 0.1, "epsilon": 0.01, "vx": 1.0, "vy": 0.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "bistable-advection", ndim)
        check_dimension_variation(result, ndim, "bistable-advection")

    def test_unsupported_dimensions(self):
        """Test that bistable-advection only supports 2D."""
        preset = get_pde_preset("bistable-advection")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
