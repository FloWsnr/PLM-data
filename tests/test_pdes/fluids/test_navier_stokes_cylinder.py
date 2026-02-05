"""Tests for Navier-Stokes flow around cylinder PDE."""

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


class TestNavierStokesCylinderPDE:
    """Tests for Navier-Stokes cylinder flow equations."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("navier-stokes-cylinder")
        meta = preset.metadata

        assert meta.name == "navier-stokes-cylinder"
        assert meta.category == "fluids"
        assert meta.num_fields == 4
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "p" in meta.field_names
        assert "S" in meta.field_names
        assert meta.supported_dimensions == [2]

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = get_pde_preset("navier-stokes-cylinder")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        params = {"nu": 0.01, "M": 0.1, "U": 1.0, "cylinder_radius": 0.05}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(
            grid,
            "default",
            {"U": 0.7, "cylinder_radius": 0.05, "cylinder_x": 0.25, "cylinder_y": 0.5},
        )

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "navier-stokes-cylinder", ndim)
        check_dimension_variation(result, ndim, "navier-stokes-cylinder")

    def test_unsupported_dimensions(self):
        """Test that navier-stokes-cylinder only supports 2D."""
        preset = get_pde_preset("navier-stokes-cylinder")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
