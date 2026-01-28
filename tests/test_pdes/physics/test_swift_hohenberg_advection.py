"""Tests for Swift-Hohenberg with Advection PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


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

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)
        preset = get_pde_preset("swift-hohenberg-advection")
        pde = preset.create_pde(
            {"r": 0.1, "g1": 0.1, "g2": 0.1, "D": 1.0, "k0": 1.0, "vx": 0.1, "vy": 0.0},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test Swift-Hohenberg with Advection works in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("swift-hohenberg-advection")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Use conservative parameters with positive quintic stabilization
        pde = preset.create_pde(
            {"r": 0.1, "g1": 0.1, "g2": 0.1, "D": 1.0, "k0": 1.0, "vx": 0.1, "vy": 0.0},
            bc,
            grid,
        )
        # Use very small initial perturbations for stability
        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.01, "high": 0.01})

        # Run very short simulation (4th order PDE is numerically stiff)
        result = pde.solve(state, t_range=0.00001, dt=0.000001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "swift-hohenberg-advection", ndim)
        check_dimension_variation(result, ndim, "swift-hohenberg-advection")

    def test_unsupported_dimensions(self):
        """Test that swift-hohenberg-advection only supports 2D."""
        preset = get_pde_preset("swift-hohenberg-advection")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
