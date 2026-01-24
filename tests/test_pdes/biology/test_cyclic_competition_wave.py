"""Tests for cyclic competition wave PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestCyclicCompetitionWavePDE:
    """Tests for the cyclic competition wave equation preset."""

    def test_registered(self):
        """Test that cyclic-competition-wave is registered."""
        assert "cyclic-competition-wave" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("cyclic-competition-wave")
        meta = preset.metadata

        assert meta.name == "cyclic-competition-wave"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("cyclic-competition-wave")
        pde = preset.create_pde(
            {"a": 0.8, "b": 1.9, "D": 0.3},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test cyclic-competition-wave simulation in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("cyclic-competition-wave")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"a": 0.8, "b": 1.9, "D": 0.3}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "cyclic-competition-wave", ndim)
        check_dimension_variation(result, ndim, "cyclic-competition-wave")

    def test_unsupported_dimensions(self):
        """Test that cyclic-competition-wave only supports 2D."""
        preset = get_pde_preset("cyclic-competition-wave")
        assert preset.metadata.supported_dimensions == [2]

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
