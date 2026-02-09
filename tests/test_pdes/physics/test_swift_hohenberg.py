"""Tests for Swift-Hohenberg pattern formation PDE."""

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


class TestSwiftHohenbergPDE:
    """Tests for Swift-Hohenberg pattern formation."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("swift-hohenberg")
        meta = preset.metadata

        assert meta.name == "swift-hohenberg"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Swift-Hohenberg works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("swift-hohenberg")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Use conservative parameters with quintic stabilization
        pde = preset.create_pde({"r": 0.1, "a": 0.1, "b": -1.0, "c": -1.0, "D": 1.0}, bc, grid)
        # Use very small initial perturbations for stability
        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.01, "high": 0.01})

        # Run very short simulation (4th order PDE is numerically stiff)
        result = pde.solve(state, t_range=0.00001, dt=0.000001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "swift-hohenberg", ndim)
        check_dimension_variation(result, ndim, "swift-hohenberg")
