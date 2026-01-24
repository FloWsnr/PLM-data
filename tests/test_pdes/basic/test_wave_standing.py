"""Tests for standing wave equation PDE."""

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


class TestStandingWavePDE:
    """Tests for the standing wave equation preset."""

    def test_registered(self):
        """Test that wave-standing is registered."""
        assert "wave-standing" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("wave-standing")
        meta = preset.metadata

        assert meta.name == "wave-standing"
        assert meta.category == "basic"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("wave-standing")
        pde = preset.create_pde(
            parameters={"D": 1.0},
            bc={"x": "periodic", "y": "periodic"},
            grid=grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test standing wave equation works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("wave-standing")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"D": 1.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "wave-standing", ndim)
        check_dimension_variation(result, ndim, "wave-standing")
