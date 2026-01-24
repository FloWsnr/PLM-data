"""Tests for 2D Navier-Stokes PDE."""

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


class TestNavierStokesPDE:
    """Tests for 2D Navier-Stokes equations."""

    def test_registered(self):
        """Test that navier-stokes is registered."""
        assert "navier-stokes" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("navier-stokes")
        meta = preset.metadata

        assert meta.name == "navier-stokes"
        assert meta.category == "fluids"
        assert meta.num_fields == 4
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "p" in meta.field_names
        assert "S" in meta.field_names
        assert meta.supported_dimensions == [2]

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("navier-stokes")
        params = {"nu": 0.01, "M": 0.1, "D": 0.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, grid)

        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = get_pde_preset("navier-stokes")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        params = {"nu": 0.01, "M": 0.1, "D": 0.0}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(grid, "shear-layer", {"amplitude": 0.5})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "navier-stokes", ndim)
        check_dimension_variation(result, ndim, "navier-stokes")

    def test_unsupported_dimensions(self):
        """Test that navier-stokes only supports 2D."""
        preset = get_pde_preset("navier-stokes")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
