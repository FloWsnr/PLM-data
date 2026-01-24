"""Tests for Cahn-Hilliard phase separation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.cahn_hilliard import CahnHilliardPDE

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


class TestCahnHilliardPDE:
    """Tests for Cahn-Hilliard phase separation."""

    def test_registered(self):
        """Test that cahn-hilliard is registered."""
        assert "cahn-hilliard" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cahn-hilliard")
        meta = preset.metadata

        assert meta.name == "cahn-hilliard"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("cahn-hilliard")
        params = preset.get_default_parameters()

        # New parameter names from reference
        assert "r" in params
        assert "a" in params
        assert "D" in params
        assert params["r"] == 0.01
        assert params["a"] == 1.0

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("cahn-hilliard")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("cahn-hilliard")
        state = preset.create_initial_state(
            small_grid, "random-uniform", {"low": -0.1, "high": 0.1}
        )

        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        # Cahn-Hilliard has 4th order terms, use very short time
        result, config = run_short_simulation("cahn-hilliard", "physics", t_end=0.0001)

        assert result is not None
        assert np.isfinite(result.data).all()
        assert config["preset"] == "cahn-hilliard"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Cahn-Hilliard works in all supported dimensions."""
        np.random.seed(42)
        preset = CahnHilliardPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.1, "high": 0.1})

        # Run short simulation (4th order PDE is stiff)
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "cahn-hilliard", ndim)
        check_dimension_variation(result, ndim, "cahn-hilliard")
