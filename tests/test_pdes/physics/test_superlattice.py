"""Tests for superlattice pattern formation PDE (coupled Brusselator + Lengyel-Epstein)."""

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


class TestSuperlatticePDE:
    """Tests for superlattice pattern formation."""

    def test_registered(self):
        """Test that superlattice is registered."""
        assert "superlattice" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("superlattice")
        meta = preset.metadata

        assert meta.name == "superlattice"
        assert meta.category == "physics"
        assert meta.num_fields == 4
        assert "u1" in meta.field_names
        assert "v1" in meta.field_names
        assert "u2" in meta.field_names
        assert "v2" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        from pde import CartesianGrid

        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("superlattice")
        pde = preset.create_pde(
            {"a": 3.0, "b": 9.0, "c": 15.0, "d": 9.0, "alpha": 0.15, "D_uone": 4.3, "D_utwo": 50.0, "D_uthree": 22.0, "D_ufour": 660.0},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Superlattice PDE works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("superlattice")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde(
            {"a": 3.0, "b": 9.0, "c": 15.0, "d": 9.0, "alpha": 0.15, "D_uone": 4.3, "D_utwo": 50.0, "D_uthree": 22.0, "D_ufour": 660.0},
            bc,
            grid,
        )
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "superlattice", ndim)
        check_dimension_variation(result, ndim, "superlattice")
