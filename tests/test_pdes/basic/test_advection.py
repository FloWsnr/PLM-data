"""Tests for advection-diffusion equation PDE."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.advection import AdvectionPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


class TestAdvectionPDE:
    """Tests for the Advection-diffusion equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("advection")
        meta = preset.metadata

        assert meta.name == "advection"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("advection")
        params = preset.get_default_parameters()

        assert "D" in params
        assert "V" in params
        assert "theta" in params
        assert "mode" in params
        assert params["D"] == 1.0
        assert params["V"] == 0.10
        assert params["mode"] == 0  # rotational

    def test_create_pde_rotational(self, small_grid):
        """Test PDE creation with rotational mode."""
        preset = get_pde_preset("advection")
        params = {"D": 1.0, "V": 0.1, "mode": 0}
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_pde_directed(self, small_grid):
        """Test PDE creation with directed mode."""
        preset = get_pde_preset("advection")
        params = {"D": 1.0, "V": 6.0, "theta": -2.0, "mode": 1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "advection" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("advection", "basic", t_end=0.1)

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "advection"

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_dimension_support(self, ndim: int):
        """Test advection equation works in supported dimensions (2D and 3D)."""
        np.random.seed(42)
        preset = AdvectionPDE()

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
        check_result_finite(result, "advection", ndim)

    def test_dimension_1d_not_supported(self):
        """Test that advection rejects 1D (implementation requires 2D+ for rotational mode)."""
        preset = AdvectionPDE()
        assert 1 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
