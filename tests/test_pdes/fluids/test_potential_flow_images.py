"""Tests for Potential Flow Images PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestPotentialFlowImagesPDE:
    """Tests for Potential Flow Images PDE."""

    def test_registered(self):
        """Test that PDE is registered."""
        assert "potential-flow-images" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("potential-flow-images")
        meta = preset.metadata

        assert meta.name == "potential-flow-images"
        assert meta.category == "fluids"
        assert meta.num_fields == 2
        assert "phi" in meta.field_names
        assert "s" in meta.field_names
        assert meta.supported_dimensions == [2]

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)
        preset = get_pde_preset("potential-flow-images")
        params = {"strength": 1.0, "wall_x": 0.5, "sigma": 0.5, "omega": 0.5, "amplitude": 0.5}
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, grid)

        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = get_pde_preset("potential-flow-images")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        params = {"strength": 10.0, "wall_x": 0.5, "sigma": 0.1, "omega": 0.5, "amplitude": 0.2}
        pde = preset.create_pde(params, bc, grid, init_params={"motion": "static"})

        state = preset.create_initial_state(grid, "source-with-image", {"strength": 10.0})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "potential-flow-images", ndim)
        check_dimension_variation(result, ndim, "potential-flow-images")

    def test_unsupported_dimensions(self):
        """Test that potential-flow-images only supports 2D."""
        preset = get_pde_preset("potential-flow-images")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
