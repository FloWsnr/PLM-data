"""Tests for Klausmeier on topography PDE."""

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


class TestKlausmeierTopographyPDE:
    """Tests for Klausmeier on topography PDE."""

    def test_registered(self):
        """Test that klausmeier-topography is registered."""
        assert "klausmeier-topography" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("klausmeier-topography")
        meta = preset.metadata

        assert meta.name == "klausmeier-topography"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "n" in meta.field_names
        assert "w" in meta.field_names
        assert "T" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("klausmeier-topography")
        pde = preset.create_pde(
            {"a": 2.0, "m": 0.54, "V": 100.0, "Dn": 1.0, "Dw": 2.0},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Klausmeier Topography works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("klausmeier-topography")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"a": 2.0, "m": 0.54, "V": 100.0, "Dn": 1.0, "Dw": 2.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "klausmeier-topography", ndim)
        check_dimension_variation(result, ndim, "klausmeier-topography")
