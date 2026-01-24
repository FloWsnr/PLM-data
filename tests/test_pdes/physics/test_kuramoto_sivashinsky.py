"""Tests for Kuramoto-Sivashinsky PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.kuramoto_sivashinsky import KuramotoSivashinskyPDE

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


class TestKuramotoSivashinskyPDE:
    """Tests for the Kuramoto-Sivashinsky equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("kuramoto-sivashinsky")
        meta = preset.metadata

        assert meta.name == "kuramoto-sivashinsky"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("kuramoto-sivashinsky")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "kuramoto-sivashinsky" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        # KS equation is fourth-order and very stiff - use tiny t_end
        result, config = run_short_simulation("kuramoto-sivashinsky", "physics", t_end=0.001)

        # Check that result is finite and valid
        assert result is not None
        assert np.isfinite(result.data).all(), "Simulation produced NaN or Inf values"
        assert config["preset"] == "kuramoto-sivashinsky"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Kuramoto-Sivashinsky works in all supported dimensions."""
        np.random.seed(42)
        preset = KuramotoSivashinskyPDE()

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

        # Run very short simulation (4th order PDE is numerically stiff)
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "kuramoto-sivashinsky", ndim)
        check_dimension_variation(result, ndim, "kuramoto-sivashinsky")
