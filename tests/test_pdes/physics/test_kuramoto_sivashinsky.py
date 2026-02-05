"""Tests for Kuramoto-Sivashinsky PDE."""

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


class TestKuramotoSivashinskyPDE:
    """Tests for the Kuramoto-Sivashinsky equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("kuramoto-sivashinsky")
        meta = preset.metadata

        assert meta.name == "kuramoto-sivashinsky"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Kuramoto-Sivashinsky works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("kuramoto-sivashinsky")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"nu": 1.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.1, "high": 0.1})

        # Run very short simulation (4th order PDE is numerically stiff)
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "kuramoto-sivashinsky", ndim)
        check_dimension_variation(result, ndim, "kuramoto-sivashinsky")
