"""Tests for Vorticity Bounded PDE."""

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


class TestVorticityBoundedPDE:
    """Tests for the Vorticity Bounded PDE."""

    def test_registered(self):
        """Test that PDE is registered."""
        assert "vorticity-bounded" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("vorticity-bounded")
        meta = preset.metadata

        assert meta.name == "vorticity-bounded"
        assert meta.category == "fluids"
        assert meta.num_fields == 3
        assert "omega" in meta.field_names
        assert "psi" in meta.field_names
        assert "S" in meta.field_names
        assert meta.supported_dimensions == [2]

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("vorticity-bounded")
        params = {"nu": 0.01, "epsilon": 0.1, "D": 0.0, "k": 0.1}
        bc = preset.get_default_bc()

        pde = preset.create_pde(params, bc, grid)

        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = get_pde_preset("vorticity-bounded")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = preset.get_default_bc()

        params = {"nu": 0.01, "epsilon": 0.1, "D": 0.0, "k": 4}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(grid, "default", {"k": 4})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "vorticity-bounded", ndim)
        check_dimension_variation(result, ndim, "vorticity-bounded")

    def test_unsupported_dimensions(self):
        """Test that vorticity-bounded only supports 2D."""
        preset = get_pde_preset("vorticity-bounded")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
