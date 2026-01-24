"""Tests for Bistable Allen-Cahn PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestBistableAllenCahnPDE:
    """Tests for Bistable Allen-Cahn PDE."""

    def test_registered(self):
        """Test that bistable-allen-cahn is registered."""
        assert "bistable-allen-cahn" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("bistable-allen-cahn")
        meta = preset.metadata

        assert meta.name == "bistable-allen-cahn"
        assert meta.category == "biology"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("bistable-allen-cahn")
        params = {"D": 0.1, "epsilon": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("bistable-allen-cahn", "biology")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "bistable-allen-cahn"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Bistable Allen-Cahn works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("bistable-allen-cahn")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        params = {"D": 0.1, "epsilon": 0.1}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "bistable-allen-cahn", ndim)
        check_dimension_variation(result, ndim, "bistable-allen-cahn")
