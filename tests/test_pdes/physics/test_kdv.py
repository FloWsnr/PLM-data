"""Tests for Korteweg-de Vries (KdV) equation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestKdVPDE:
    """Tests for Korteweg-de Vries equation."""

    def test_registered(self):
        """Test that kdv is registered."""
        assert "kdv" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("kdv")
        meta = preset.metadata

        assert meta.name == "kdv"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        from pde import CartesianGrid

        grid = CartesianGrid([[0, 10]], [64], periodic=True)
        preset = get_pde_preset("kdv")
        pde = preset.create_pde(
            {"b": 0.0001},
            {"x-": "periodic", "x+": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1])
    def test_short_simulation(self, ndim: int):
        """Test KdV simulation in 1D."""
        np.random.seed(42)
        preset = get_pde_preset("kdv")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # KdV needs higher resolution
        grid = create_grid_for_dimension(ndim, resolution=64)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"b": 0.0001}, bc, grid)
        # IMPORTANT: Use soliton IC, not random-uniform (KdV is unstable with random data)
        state = preset.create_initial_state(grid, "soliton", {"k": 0.5, "x0": 0.25})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "kdv", ndim)
        check_dimension_variation(result, ndim, "kdv")

    def test_unsupported_dimensions(self):
        """Test that KdV only supports 1D."""
        preset = get_pde_preset("kdv")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(2)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
