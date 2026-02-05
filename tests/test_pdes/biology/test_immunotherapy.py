"""Tests for tumor-immune interaction model."""

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


class TestImmunotherapyPDE:
    """Tests for tumor-immune interaction model."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("immunotherapy")
        meta = preset.metadata

        assert meta.name == "immunotherapy"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Immunotherapy works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("immunotherapy")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde(
            {
                "Du": 0.5,
                "Dv": 0.008,
                "Dw": 4.0,
                "c": 0.3,
                "mu": 0.167,
                "p_u": 0.69167,
                "g_v": 0.1,
                "p_v": 1.0,
                "p_w": 27.778,
                "g_w": 0.001,
                "nu": 55.55556,
                "s_u": 0.0,
                "s_w": 10.0,
            },
            bc,
            grid,
        )
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "immunotherapy", ndim)
        check_dimension_variation(result, ndim, "immunotherapy")
