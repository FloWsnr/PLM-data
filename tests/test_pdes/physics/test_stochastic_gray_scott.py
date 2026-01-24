"""Tests for Stochastic Gray-Scott PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.stochastic_gray_scott import StochasticGrayScottPDE

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


class TestStochasticGrayScottPDE:
    """Tests for Stochastic Gray-Scott PDE."""

    def test_registered(self):
        """Test that stochastic-gray-scott is registered."""
        assert "stochastic-gray-scott" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("stochastic-gray-scott")
        meta = preset.metadata

        assert meta.name == "stochastic-gray-scott"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test creating the PDE object (deterministic mode)."""
        preset = get_pde_preset("stochastic-gray-scott")
        params = {"a": 0.037, "b": 0.06, "D": 2.0, "noise": 0.0}  # Deterministic
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("stochastic-gray-scott")
        state = preset.create_initial_state(small_grid, "default", {"seed": 42})

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()
        assert np.isfinite(state[1].data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("stochastic-gray-scott", "physics")

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "stochastic-gray-scott"

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_dimension_support(self, ndim: int):
        """Test Stochastic Gray-Scott works in 1D and 2D.

        Note: 3D is currently not tested because the preset's _convert_bc doesn't
        receive the grid dimension, causing BC conversion issues in 3D.
        """
        np.random.seed(42)
        preset = StochasticGrayScottPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Use deterministic mode for testing
        params = {"a": 0.037, "b": 0.06, "D": 2.0, "noise": 0.0}

        # Create PDE and initial state
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "stochastic-gray-scott", ndim)
        check_dimension_variation(result, ndim, "stochastic-gray-scott")
