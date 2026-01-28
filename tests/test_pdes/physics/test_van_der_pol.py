"""Tests for Van der Pol Oscillator PDE."""

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


class TestVanDerPolPDE:
    """Tests for Van der Pol Oscillator PDE."""

    def test_registered(self):
        """Test that van-der-pol is registered."""
        assert "van-der-pol" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("van-der-pol")
        meta = preset.metadata

        assert meta.name == "van-der-pol"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "X" in meta.field_names
        assert "Y" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)
        preset = get_pde_preset("van-der-pol")
        pde = preset.create_pde(
            {"epsilon": 1.0, "mu": 0.1, "D": 0.01},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Van der Pol works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("van-der-pol")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"epsilon": 1.0, "mu": 0.1, "D": 0.01}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "van-der-pol", ndim)
        check_dimension_variation(result, ndim, "van-der-pol")
