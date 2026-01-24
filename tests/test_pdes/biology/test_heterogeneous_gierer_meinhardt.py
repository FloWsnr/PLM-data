"""Tests for Heterogeneous Gierer-Meinhardt PDE."""

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


class TestHeterogeneousGiererMeinhardtPDE:
    """Tests for Heterogeneous Gierer-Meinhardt PDE."""

    def test_registered(self):
        """Test that heterogeneous-gierer-meinhardt is registered."""
        assert "heterogeneous-gierer-meinhardt" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("heterogeneous-gierer-meinhardt")
        meta = preset.metadata

        assert meta.name == "heterogeneous-gierer-meinhardt"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "X" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("heterogeneous-gierer-meinhardt")
        pde = preset.create_pde(
            {"D": 55.0, "a": 1.0, "b": 1.5, "c": 6.1, "A": 0.0, "B": 0.0},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Heterogeneous Gierer-Meinhardt works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("heterogeneous-gierer-meinhardt")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"D": 55.0, "a": 1.0, "b": 1.5, "c": 6.1, "A": 0.0, "B": 0.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "heterogeneous-gierer-meinhardt", ndim)
        check_dimension_variation(result, ndim, "heterogeneous-gierer-meinhardt")
