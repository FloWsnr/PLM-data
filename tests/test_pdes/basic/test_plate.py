"""Tests for plate vibration equation PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestPlatePDE:
    """Tests for the plate vibration equation preset."""

    def test_registered(self):
        """Test that plate PDE is registered."""
        assert "plate" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("plate")
        meta = preset.metadata

        assert meta.name == "plate"
        assert meta.category == "basic"
        assert meta.num_fields == 3  # u, v, w
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)
        preset = get_pde_preset("plate")
        pde = preset.create_pde(
            parameters={"D": 1.0, "Q": 10.0, "C": 0.1, "D_c": 0.1},
            bc={"x": "neumann", "y": "neumann"},
            grid=grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_short_simulation(self, ndim: int):
        """Test plate equation works in supported dimensions (1D and 2D)."""
        np.random.seed(42)
        preset = get_pde_preset("plate")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Use non-periodic for plate equation
        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        params = {"D": 1.0, "Q": 10.0, "C": 0.1, "D_c": 0.1}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": -5.0, "high": -3.0}, parameters=params, bc=bc)

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "plate", ndim)
        check_dimension_variation(result, ndim, "plate")

    def test_unsupported_dimensions(self):
        """Test that plate rejects 3D."""
        preset = get_pde_preset("plate")
        assert 3 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
