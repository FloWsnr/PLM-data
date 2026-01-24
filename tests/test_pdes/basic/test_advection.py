"""Tests for advection-diffusion equation PDEs."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.advection import AdvectionPDE
from pde_sim.pdes.basic.advection_rotational import AdvectionRotationalPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestAdvectionPDE:
    """Tests for the uniform flow advection-diffusion equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("advection")
        meta = preset.metadata

        assert meta.name == "advection"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation with velocity components."""
        preset = get_pde_preset("advection")
        params = {"D": 1.0, "vx": 5.0, "vy": 2.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "advection" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("advection", "basic")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "advection"

    def test_dimension_2d_support(self):
        """Test advection equation works in 2D."""
        np.random.seed(42)
        preset = AdvectionPDE()

        # Check dimension is supported
        assert 2 in preset.metadata.supported_dimensions
        preset.validate_dimension(2)

        # Create grid and BCs
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        # Create PDE and initial state
        params = {"D": 0.0, "vx": 1.0, "vy": 0.5}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "advection", 2)
        check_dimension_variation(result, 2, "advection")

    def test_dimension_1d_not_supported(self):
        """Test that advection rejects 1D (uses d_dy)."""
        preset = AdvectionPDE()
        assert 1 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)

    def test_dimension_3d_not_supported(self):
        """Test that advection rejects 3D."""
        preset = AdvectionPDE()
        assert 3 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)


class TestAdvectionRotationalPDE:
    """Tests for the rotational (vortex) advection-diffusion equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("advection-rotational")
        meta = preset.metadata

        assert meta.name == "advection-rotational"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation with angular velocity."""
        preset = get_pde_preset("advection-rotational")
        params = {"D": 1.0, "omega": 0.5}
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "advection-rotational" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("advection-rotational", "basic")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "advection-rotational"

    def test_dimension_2d_support(self):
        """Test advection-rotational equation works in 2D."""
        np.random.seed(42)
        preset = AdvectionRotationalPDE()

        # Check dimension is supported
        assert 2 in preset.metadata.supported_dimensions
        preset.validate_dimension(2)

        # Create grid and BCs (non-periodic for rotational flow)
        grid = create_grid_for_dimension(2, resolution=16, periodic=False)
        bc = create_bc_for_dimension(2, periodic=False)

        # Create PDE and initial state
        params = {"D": 0.0, "omega": 1.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "advection-rotational", 2)
        check_dimension_variation(result, 2, "advection-rotational")

    def test_dimension_1d_not_supported(self):
        """Test that advection-rotational rejects 1D (uses x, y coordinates)."""
        preset = AdvectionRotationalPDE()
        assert 1 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)

    def test_dimension_3d_not_supported(self):
        """Test that advection-rotational rejects 3D."""
        preset = AdvectionRotationalPDE()
        assert 3 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)

    def test_clockwise_vs_counterclockwise(self, small_grid):
        """Test that positive and negative omega give different results."""
        preset = AdvectionRotationalPDE()
        bc = {"x": "dirichlet", "y": "dirichlet"}

        # Create PDE with positive omega (counterclockwise)
        pde_ccw = preset.create_pde({"D": 0.1, "omega": 1.0}, bc, small_grid)

        # Create PDE with negative omega (clockwise)
        pde_cw = preset.create_pde({"D": 0.1, "omega": -1.0}, bc, small_grid)

        # Both should be valid PDEs
        assert pde_ccw is not None
        assert pde_cw is not None
