"""Tests for Fokker-Planck equation PDE preset."""

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


class TestFokkerPlanckPDE:
    """Tests for Fokker-Planck PDE."""

    def test_registered(self):
        """Test that fokker-planck is registered."""
        assert "fokker-planck" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fokker-planck")
        meta = preset.metadata

        assert meta.name == "fokker-planck"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "p" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        from pde import CartesianGrid

        grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=False)
        preset = get_pde_preset("fokker-planck")
        pde = preset.create_pde(
            {"gamma": 1.0, "D": 1.0, "k": 1.0, "T": 1.0},
            {"x-": "neumann:0", "x+": "neumann:0", "y-": "neumann:0", "y+": "neumann:0"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test Fokker-Planck simulation in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("fokker-planck")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Use non-periodic BCs for Fokker-Planck
        grid = create_grid_for_dimension(ndim, resolution=16, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        pde = preset.create_pde({"gamma": 1.0, "D": 1.0, "k": 1.0, "T": 1.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "fokker-planck", ndim)
        check_dimension_variation(result, ndim, "fokker-planck")

    def test_unsupported_dimensions(self):
        """Test that fokker-planck only supports 2D."""
        preset = get_pde_preset("fokker-planck")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
