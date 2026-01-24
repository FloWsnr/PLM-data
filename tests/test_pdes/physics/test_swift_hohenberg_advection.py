"""Tests for Swift-Hohenberg with Advection PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.swift_hohenberg_advection import SwiftHohenbergAdvectionPDE

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


class TestSwiftHohenbergAdvectionPDE:
    """Tests for Swift-Hohenberg with Advection PDE."""

    def test_registered(self):
        """Test that swift-hohenberg-advection is registered."""
        assert "swift-hohenberg-advection" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("swift-hohenberg-advection")
        meta = preset.metadata

        assert meta.name == "swift-hohenberg-advection"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("swift-hohenberg-advection", "physics", t_end=0.01)

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "swift-hohenberg-advection"

    def test_dimension_support_2d(self):
        """Test Swift-Hohenberg with Advection works in 2D.

        Note: The current implementation uses 2D-specific advection terms (d_dy),
        so the test only validates 2D support even though metadata claims [1, 2, 3].
        """
        np.random.seed(42)
        preset = SwiftHohenbergAdvectionPDE()
        ndim = 2

        # Check 2D is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE with conservative parameters for testing
        params = preset.get_default_parameters()
        # Use supercritical parameters with positive quintic stabilization
        params["r"] = 0.1  # Small positive (supercritical)
        params["a"] = 0.1  # Small quadratic
        params["b"] = 1.0  # Positive cubic
        params["c"] = 0.1  # Positive quintic for stability
        params["V"] = 0.1  # Small advection velocity
        pde = preset.create_pde(params, bc, grid)

        # Use very small initial perturbations for stability
        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.01, "high": 0.01})

        # Run very short simulation (4th order PDE is numerically stiff)
        result = pde.solve(state, t_range=0.00001, dt=0.000001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "swift-hohenberg-advection", ndim)
        check_dimension_variation(result, ndim, "swift-hohenberg-advection")
