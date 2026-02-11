"""Tests for Phase-Field Crystal model."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset
from pde_sim.core.config import BoundaryConfig
from tests.test_pdes.dimension_test_helpers import (
    check_result_finite,
    check_dimension_variation,
)


class TestPhaseFieldCrystalPDE:
    """Tests for Phase-Field Crystal model."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("phase-field-crystal")
        meta = preset.metadata

        assert meta.name == "phase-field-crystal"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Phase-Field Crystal works in all supported dimensions.

        Uses larger domain ([0,20]) to keep dx large enough for stability
        with the 6th-order operator (dx^6 must not be too small).
        """
        np.random.seed(42)
        preset = get_pde_preset("phase-field-crystal")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Use [0, 20] domain with 16 points → dx = 1.25
        # Standard [0, 1] with 16 pts → dx = 0.0625, dx^6 ≈ 6e-8 → unstable
        resolution = 8 if ndim == 3 else 16
        bounds = [[0, 20]] * ndim
        res = [resolution] * ndim
        periodic = [True] * ndim
        grid = CartesianGrid(bounds, res, periodic=periodic)

        if ndim == 1:
            bc = BoundaryConfig(x_minus="periodic", x_plus="periodic")
        elif ndim == 2:
            bc = BoundaryConfig(x_minus="periodic", x_plus="periodic", y_minus="periodic", y_plus="periodic")
        else:
            bc = BoundaryConfig(x_minus="periodic", x_plus="periodic", y_minus="periodic", y_plus="periodic", z_minus="periodic", z_plus="periodic")

        pde = preset.create_pde({"epsilon": 0.325}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.3})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "phase-field-crystal", ndim)
        check_dimension_variation(result, ndim, "phase-field-crystal")

    def test_custom_ic(self):
        """Test the default crystallization IC."""
        preset = get_pde_preset("phase-field-crystal")
        grid = CartesianGrid([[0, 20], [0, 20]], [16, 16], periodic=[True, True])

        state = preset.create_initial_state(
            grid, "phase-field-crystal-default",
            {"mean_density": 0.2, "amplitude": 0.05, "seed": 42},
        )
        assert isinstance(state, ScalarField)
        # Should be centered around mean_density
        assert abs(state.data.mean() - 0.2) < 0.1
