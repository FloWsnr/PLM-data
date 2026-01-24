"""Tests for Swift-Hohenberg pattern formation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.swift_hohenberg import SwiftHohenbergPDE

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


class TestSwiftHohenbergPDE:
    """Tests for Swift-Hohenberg pattern formation."""

    def test_registered(self):
        """Test that swift-hohenberg is registered."""
        assert "swift-hohenberg" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("swift-hohenberg")
        meta = preset.metadata

        assert meta.name == "swift-hohenberg"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("swift-hohenberg")
        params = preset.get_default_parameters()

        assert "r" in params
        assert "a" in params
        assert "b" in params
        assert "c" in params
        assert "D" in params

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("swift-hohenberg")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("swift-hohenberg")
        state = preset.create_initial_state(
            small_grid, "random-uniform", {"low": -0.1, "high": 0.1}
        )

        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config.

        Swift-Hohenberg has 4th order terms making it numerically stiff.
        """
        # Run a very short simulation (fourth-order PDE is stiff)
        result, config = run_short_simulation("swift-hohenberg", "physics")

        # Check that result is finite and valid
        assert result is not None
        assert np.isfinite(result.data).all(), "Simulation produced NaN or Inf values"
        assert config["preset"] == "swift-hohenberg"

    def test_subcritical_with_quintic(self, small_grid):
        """Test subcritical regime with quintic term.

        Parameters for subcritical localised patterns: r<0, a>0, b<0, c<0.
        """
        # Use larger domain for stability
        sh_grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)

        preset = get_pde_preset("swift-hohenberg")
        params = {"r": -0.1, "a": 0.5, "b": -1.0, "c": -0.1, "D": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, sh_grid)
        state = preset.create_initial_state(
            sh_grid, "random-uniform", {"low": -0.05, "high": 0.05, "seed": 42}
        )

        # Run a very short simulation with tiny timestep (fourth-order PDE is stiff)
        result = pde.solve(state, t_range=0.001, dt=1e-6)

        # Check that result is finite and valid
        assert result is not None
        assert np.isfinite(result.data).all(), "Simulation produced NaN or Inf values"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Swift-Hohenberg works in all supported dimensions."""
        np.random.seed(42)
        preset = SwiftHohenbergPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs - use larger domain for stability
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE with conservative parameters for testing
        params = preset.get_default_parameters()
        # Use supercritical parameters with positive quintic stabilization
        params["r"] = 0.1  # Small positive (supercritical)
        params["a"] = 0.1  # Small quadratic
        params["b"] = 1.0  # Positive cubic
        params["c"] = 0.1  # Positive quintic for stability
        pde = preset.create_pde(params, bc, grid)

        # Use very small initial perturbations for stability
        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.01, "high": 0.01})

        # Run very short simulation (4th order PDE is numerically stiff)
        result = pde.solve(state, t_range=0.00001, dt=0.000001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "swift-hohenberg", ndim)
        check_dimension_variation(result, ndim, "swift-hohenberg")
