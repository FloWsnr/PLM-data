"""Tests for Complex Ginzburg-Landau PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestComplexGinzburgLandauPDE:
    """Tests for Complex Ginzburg-Landau PDE."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("complex-ginzburg-landau")
        meta = preset.metadata

        assert meta.name == "complex-ginzburg-landau"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Complex Ginzburg-Landau works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("complex-ginzburg-landau")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde(
            {"c1": 0.5, "c3": 0.5, "D": 1.0, "alpha": 0.0, "mu": 1.0, "omega": 0.0, "forcing_A": 0.0, "forcing_omega": 0.0},
            bc,
            grid,
        )
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "complex-ginzburg-landau", ndim)
        check_dimension_variation(result, ndim, "complex-ginzburg-landau")
