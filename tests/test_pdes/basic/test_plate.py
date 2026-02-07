"""Tests for plate vibration equation PDE."""

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


class TestPlatePDE:
    """Tests for the plate vibration equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("plate")
        meta = preset.metadata

        assert meta.name == "plate"
        assert meta.category == "basic"
        assert meta.num_fields == 3  # u, v, w
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_short_simulation(self, ndim: int):
        """Test plate equation works in supported dimensions (1D and 2D)."""
        np.random.seed(42)
        preset = get_pde_preset("plate")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Use non-periodic for plate equation
        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        params = {"D": 1.0, "Q": 10.0, "C": 0.1, "D_c": 0.1}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": -5.0, "high": -3.0}, parameters=params, bc=bc)

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "plate", ndim)
        check_dimension_variation(result, ndim, "plate")

    def test_w_field_evolves(self):
        """Test that w field evolves and tracks D * laplace(u)."""
        np.random.seed(42)
        preset = get_pde_preset("plate")

        grid = create_grid_for_dimension(2, resolution=16, periodic=False)
        bc = create_bc_for_dimension(2, periodic=False)

        params = {"D": 1.0, "Q": 10.0, "C": 0.1, "D_c": 0.1}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(
            grid, "random-uniform", {"low": -5.0, "high": -3.0},
            parameters=params, bc=bc,
        )

        w_initial = state[2].data.copy()

        result = pde.solve(
            state, t_range=0.001, dt=0.0001, solver="euler",
            tracker=None, backend="numpy",
        )

        u_final, v_final, w_final = result

        # w should have changed from its initial value
        assert not np.allclose(w_final.data, w_initial), (
            "w field is stagnant — dw/dt is likely still zero"
        )

        # w should approximately equal D * laplace(u) at the final state
        bc_spec = preset._convert_bc(bc)
        expected_w = params["D"] * u_final.laplace(bc=bc_spec).data
        np.testing.assert_allclose(
            w_final.data, expected_w, rtol=0.1, atol=1e-6,
            err_msg="w does not track D * laplace(u)",
        )

    def test_unsupported_dimensions(self):
        """Test that plate rejects 3D."""
        preset = get_pde_preset("plate")
        assert 3 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
