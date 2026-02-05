"""Tests for Potential Flow Dipoles PDE."""

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


class TestPotentialFlowDipolesPDE:
    """Tests for Potential Flow Dipoles PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("potential-flow-dipoles")
        meta = preset.metadata

        assert meta.name == "potential-flow-dipoles"
        assert meta.category == "fluids"
        assert meta.num_fields == 1
        assert "phi" in meta.field_names
        assert meta.supported_dimensions == [2]

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = get_pde_preset("potential-flow-dipoles")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        params = {"strength": 1.0, "separation": 0.2, "sigma": 0.1, "omega": 1.0, "orbit_radius": 0.2}
        pde = preset.create_pde(params, bc, grid, init_params={"motion": "circular"})

        state = preset.create_initial_state(grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "potential-flow-dipoles", ndim)
        check_dimension_variation(result, ndim, "potential-flow-dipoles")

    def test_unsupported_dimensions(self):
        """Test that potential-flow-dipoles only supports 2D."""
        preset = get_pde_preset("potential-flow-dipoles")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
