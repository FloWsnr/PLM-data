"""Tests for advection-diffusion equation PDE."""

import numpy as np
import pytest
from pde import ScalarField

from pde_sim.pdes import get_pde_preset
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestAdvectionPDE:
    """Tests for the uniform flow advection-diffusion equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("advection")
        meta = preset.metadata

        assert meta.name == "advection"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test advection equation works in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("advection")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"D": 0.0, "vx": 1.0, "vy": 0.5}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "advection", ndim)
        check_dimension_variation(result, ndim, "advection")

    def test_unsupported_dimensions(self):
        """Test that advection rejects 1D and 3D."""
        preset = get_pde_preset("advection")
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
