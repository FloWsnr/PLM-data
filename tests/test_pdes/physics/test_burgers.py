"""Tests for Burgers' equation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.burgers import BurgersPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestBurgersPDE:
    """Tests for Burgers' equation."""

    def test_registered(self):
        """Test that burgers is registered."""
        assert "burgers" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("burgers")
        meta = preset.metadata

        assert meta.name == "burgers"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("burgers")
        params = preset.get_default_parameters()

        # New parameter name from reference
        assert "epsilon" in params
        assert params["epsilon"] == 0.05

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("burgers")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("burgers")
        state = preset.create_initial_state(
            small_grid, "sine", {"wavelength": 0.5}
        )

        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("burgers", "physics", t_end=0.01)

        assert result is not None
        assert np.isfinite(result.data).all()
        assert config["preset"] == "burgers"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Burgers equation works in all supported dimensions."""
        np.random.seed(42)
        preset = BurgersPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "burgers", ndim)
