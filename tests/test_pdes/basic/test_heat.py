"""Tests for heat equation PDEs."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestHeatPDE:
    """Tests for the Heat equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("heat")
        meta = preset.metadata

        assert meta.name == "heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "T" in meta.field_names
        assert len(meta.parameters) == 1
        assert meta.parameters[0].name == "D_T"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test heat equation works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("heat")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"D_T": 1.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "heat", ndim)
        check_dimension_variation(result, ndim, "heat")


class TestInhomogeneousHeatPDE:
    """Tests for the Inhomogeneous Heat equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("inhomogeneous-heat")
        meta = preset.metadata

        assert meta.name == "inhomogeneous-heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert len(meta.parameters) == 3  # D, n, m

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test inhomogeneous-heat works in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("inhomogeneous-heat")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        pde = preset.create_pde({"D": 0.1, "n": 2, "m": 2}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "inhomogeneous-heat", ndim)
        check_dimension_variation(result, ndim, "inhomogeneous-heat")

    def test_unsupported_dimensions(self):
        """Test that inhomogeneous-heat rejects 1D and 3D."""
        preset = get_pde_preset("inhomogeneous-heat")
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)


class TestInhomogeneousDiffusionHeatPDE:
    """Tests for the Inhomogeneous Diffusion Heat equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        meta = preset.metadata

        assert meta.name == "inhomogeneous-diffusion-heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert len(meta.parameters) == 3  # D, E, n

    def test_create_pde(self):
        """Test creating PDE with spatially varying diffusion."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        pde = preset.create_pde(
            parameters={"D": 1.0, "E": 0.5, "n": 10},
            bc={"x": "dirichlet", "y": "dirichlet"},
            grid=grid,
        )
        assert pde is not None
        # Check that g, dg_dx, dg_dy are in consts
        assert "g" in pde.consts
        assert "dg_dx" in pde.consts
        assert "dg_dy" in pde.consts

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test inhomogeneous-diffusion-heat works in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("inhomogeneous-diffusion-heat")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        pde = preset.create_pde({"D": 0.1, "E": 0.5, "n": 5}, bc, grid)
        # Use uniform initial condition
        state = ScalarField.from_expression(grid, "1.0")
        state.label = "T"

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "inhomogeneous-diffusion-heat", ndim)

    def test_unsupported_dimensions(self):
        """Test that inhomogeneous-diffusion-heat rejects 1D and 3D."""
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)


class TestBlobDiffusionHeatPDE:
    """Tests for the blob diffusion heat equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("blob-diffusion-heat")
        meta = preset.metadata

        assert meta.name == "blob-diffusion-heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "T" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation with blob-based diffusion."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)
        preset = get_pde_preset("blob-diffusion-heat")
        pde = preset.create_pde(
            parameters={"D_min": 0.1, "D_max": 1.0, "n_blobs": 5, "sigma": 0.1, "seed": 42},
            bc={"x": "neumann", "y": "neumann"},
            grid=grid,
        )
        assert pde is not None
        # Check that g, dg_dx, dg_dy are in consts
        assert "g" in pde.consts
        assert "dg_dx" in pde.consts
        assert "dg_dy" in pde.consts

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test blob-diffusion-heat works in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("blob-diffusion-heat")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        pde = preset.create_pde({"D_min": 0.1, "D_max": 1.0, "n_blobs": 3, "sigma": 0.1, "seed": 42}, bc, grid)
        # Use uniform initial condition
        state = ScalarField.from_expression(grid, "1.0")
        state.label = "T"

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "blob-diffusion-heat", ndim)

    def test_unsupported_dimensions(self):
        """Test that blob-diffusion-heat rejects 1D and 3D."""
        preset = get_pde_preset("blob-diffusion-heat")
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
