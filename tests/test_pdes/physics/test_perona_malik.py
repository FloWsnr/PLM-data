"""Tests for Perona-Malik edge-preserving diffusion PDE."""

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


class TestPeronaMalikPDE:
    """Tests for Perona-Malik edge-preserving diffusion."""

    def test_registered(self):
        """Test that perona-malik is registered."""
        assert "perona-malik" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("perona-malik")
        meta = preset.metadata

        assert meta.name == "perona-malik"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_create_pde(self):
        """Test PDE creation."""
        from pde import CartesianGrid

        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("perona-malik")
        pde = preset.create_pde(
            {"K": 10.0, "dt_mult": 0.25},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test Perona-Malik simulation in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("perona-malik")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"K": 10.0, "dt_mult": 0.25}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "perona-malik", ndim)
        check_dimension_variation(result, ndim, "perona-malik")

    def test_unsupported_dimensions(self):
        """Test that perona-malik only supports 2D."""
        preset = get_pde_preset("perona-malik")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
