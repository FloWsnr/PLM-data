"""Tests for Hyperbolic Brusselator PDE."""

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


class TestHyperbolicBrusselatorPDE:
    """Tests for Hyperbolic Brusselator PDE."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("hyperbolic-brusselator")
        meta = preset.metadata

        assert meta.name == "hyperbolic-brusselator"
        assert meta.category == "biology"
        assert meta.num_fields == 4
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names
        assert "q" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Hyperbolic Brusselator works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("hyperbolic-brusselator")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde(
            {"tau": 1.0, "Du": 2.0, "Dv": 1.0, "a": 5.0, "b": 9.0, "eps": 0.001},
            bc,
            grid,
        )
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "hyperbolic-brusselator", ndim)
        check_dimension_variation(result, ndim, "hyperbolic-brusselator")
