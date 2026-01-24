"""Tests for diffusively coupled Lorenz system."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.lorenz import LorenzPDE

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


class TestLorenzPDE:
    """Tests for diffusively coupled Lorenz system."""

    def test_registered(self):
        """Test that lorenz is registered."""
        assert "lorenz" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("lorenz")
        meta = preset.metadata

        assert meta.name == "lorenz"
        assert meta.category == "physics"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"X", "Y", "Z"}

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("lorenz")
        params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0, "D": 0.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.1})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self):
        """Test running a short simulation with the Lorenz PDE using default config.

        Field names have been changed from (x, y, z) to (X, Y, Z) to avoid
        collision with 2D grid coordinate names.
        """
        result, config = run_short_simulation("lorenz", "physics")

        assert isinstance(result, FieldCollection)
        assert len(result) == 3
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert np.isfinite(result[2].data).all()
        assert config["preset"] == "lorenz"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Lorenz PDE works in all supported dimensions."""
        np.random.seed(42)
        preset = LorenzPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0, "D": 0.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "lorenz", ndim)
        check_dimension_variation(result, ndim, "lorenz")
