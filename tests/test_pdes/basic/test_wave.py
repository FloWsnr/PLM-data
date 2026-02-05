"""Tests for wave equation PDEs."""

import numpy as np
import pytest
from pde import FieldCollection

from pde_sim.pdes import get_pde_preset
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestWavePDE:
    """Tests for the Wave equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("wave")
        meta = preset.metadata

        assert meta.name == "wave"
        assert meta.category == "basic"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test wave equation works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("wave")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"D": 1.0, "C": 0.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "wave", ndim)
        check_dimension_variation(result, ndim, "wave")


class TestInhomogeneousWavePDE:
    """Tests for the inhomogeneous wave equation with spatially varying wave speed."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("inhomogeneous-wave")
        meta = preset.metadata

        assert meta.name == "inhomogeneous-wave"
        assert meta.category == "basic"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test inhomogeneous-wave works in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("inhomogeneous-wave")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        pde = preset.create_pde({"D": 1.0, "E": 0.5, "m": 4, "n": 4, "C": 0.0}, bc, grid)
        state = preset.create_initial_state(
            grid, "gaussian-blob", {"num_blobs": 1, "positions": [[0.5, 0.5]]}
        )

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        check_result_finite(result, "inhomogeneous-wave", ndim)
        check_dimension_variation(result, ndim, "inhomogeneous-wave")

    def test_unsupported_dimensions(self):
        """Test that inhomogeneous-wave rejects 1D and 3D."""
        preset = get_pde_preset("inhomogeneous-wave")
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
