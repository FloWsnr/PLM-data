"""Tests for Schrodinger equation PDE."""

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


class TestSchrodingerPDE:
    """Tests for the Schrodinger equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("schrodinger")
        meta = preset.metadata

        assert meta.name == "schrodinger"
        assert meta.category == "basic"
        assert meta.num_fields == 2  # u (real), v (imaginary)
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Schrodinger equation works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("schrodinger")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Use non-periodic for Schrodinger (particle in a box)
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        pde = preset.create_pde({"D": 1.0, "C": 1.0, "potential_type": "none", "V_strength": 0.0, "pot_n": 1, "pot_m": 1}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "schrodinger", ndim)
        check_dimension_variation(result, ndim, "schrodinger")
