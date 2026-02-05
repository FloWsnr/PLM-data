"""Tests for Zakharov-Kuznetsov PDE (2D version of KdV)."""

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


class TestZakharovKuznetsovPDE:
    """Tests for Zakharov-Kuznetsov PDE."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("zakharov-kuznetsov")
        meta = preset.metadata

        assert meta.name == "zakharov-kuznetsov"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test Zakharov-Kuznetsov simulation in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("zakharov-kuznetsov")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"alpha": 1.0, "beta": 1.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "zakharov-kuznetsov", ndim)
        check_dimension_variation(result, ndim, "zakharov-kuznetsov")

    def test_unsupported_dimensions(self):
        """Test that zakharov-kuznetsov only supports 2D."""
        preset = get_pde_preset("zakharov-kuznetsov")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
