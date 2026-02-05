"""Tests for Thermal Convection PDE."""

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


class TestThermalConvectionPDE:
    """Tests for Thermal Convection PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("thermal-convection")
        meta = preset.metadata

        assert meta.name == "thermal-convection"
        assert meta.category == "fluids"
        assert meta.num_fields == 3
        assert "omega" in meta.field_names
        assert "psi" in meta.field_names
        assert "b" in meta.field_names
        assert meta.supported_dimensions == [2]

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = get_pde_preset("thermal-convection")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        params = {"nu": 0.01, "epsilon": 0.1, "kappa": 0.01, "T_b": 1.0}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(grid, "default", {"noise": 0.01, "seed": 42})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "thermal-convection", ndim)
        check_dimension_variation(result, ndim, "thermal-convection")

    def test_unsupported_dimensions(self):
        """Test that thermal-convection only supports 2D."""
        preset = get_pde_preset("thermal-convection")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
