"""Tests for Porous Medium Equation."""

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


class TestPorousMediumPDE:
    """Tests for Porous Medium Equation."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("porous-medium")
        meta = preset.metadata

        assert meta.name == "porous-medium"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Porous Medium Equation works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("porous-medium")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"m": 2.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "porous-medium", ndim)
        check_dimension_variation(result, ndim, "porous-medium")

    def test_non_negative_clipping(self):
        """Test that generic ICs are clipped to non-negative values."""
        preset = get_pde_preset("porous-medium")
        grid = create_grid_for_dimension(1, resolution=16)

        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.5, "high": 0.5})
        assert (state.data >= 0.0).all(), "Porous medium IC should be non-negative"

    def test_custom_ic(self):
        """Test the Barenblatt-like default IC."""
        preset = get_pde_preset("porous-medium")
        grid = create_grid_for_dimension(2, resolution=16)

        state = preset.create_initial_state(
            grid, "porous-medium-default",
            {"amplitude": 1.0, "radius": 0.3, "seed": 42},
        )
        assert isinstance(state, ScalarField)
        assert (state.data >= 0.0).all(), "Default IC should be non-negative"
        # Center should have highest values
        assert state.data[8, 8] > state.data[0, 0]
