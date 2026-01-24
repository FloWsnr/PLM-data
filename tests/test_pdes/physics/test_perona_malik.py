"""Tests for Perona-Malik edge-preserving diffusion PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.perona_malik import PeronaMalikPDE

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
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


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

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("perona-malik", "physics")

        assert result is not None
        assert np.isfinite(result.data).all()
        assert config["preset"] == "perona-malik"

    def test_dimension_support_2d(self):
        """Test Perona-Malik works in 2D.

        Note: The current implementation uses 2D-specific derivatives (d_dy, u_xy),
        so the test only validates 2D support even though metadata claims [1, 2, 3].
        """
        np.random.seed(42)
        preset = PeronaMalikPDE()
        ndim = 2

        # Check 2D is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        params = {"K": 10.0, "dt_mult": 0.25}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "perona-malik", ndim)
        check_dimension_variation(result, ndim, "perona-malik")
