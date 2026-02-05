"""Tests for FitzHugh-Nagumo 3-species PDE."""

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


class TestFitzHughNagumo3PDE:
    """Tests for the FitzHugh-Nagumo 3-species equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("fitzhugh-nagumo-3")
        meta = preset.metadata

        assert meta.name == "fitzhugh-nagumo-3"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test fitzhugh-nagumo-3 simulation in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("fitzhugh-nagumo-3")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde(
            {"Dv": 40.0, "Dw": 200.0, "a_v": 0.2, "e_v": 0.2, "e_w": 1.0, "a_w": 0.5, "a_z": -0.1},
            bc,
            grid,
        )
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "fitzhugh-nagumo-3", ndim)
        check_dimension_variation(result, ndim, "fitzhugh-nagumo-3")

    def test_unsupported_dimensions(self):
        """Test that fitzhugh-nagumo-3 only supports 2D."""
        preset = get_pde_preset("fitzhugh-nagumo-3")
        assert preset.metadata.supported_dimensions == [2]

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
