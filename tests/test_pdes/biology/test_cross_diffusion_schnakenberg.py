"""Tests for Cross-Diffusion Schnakenberg PDE."""

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


class TestCrossDiffusionSchnakenbergPDE:
    """Tests for Cross-Diffusion Schnakenberg PDE."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cross-diffusion-schnakenberg")
        meta = preset.metadata

        assert meta.name == "cross-diffusion-schnakenberg"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Cross-Diffusion Schnakenberg works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("cross-diffusion-schnakenberg")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde(
            {"Duu": 1.0, "Duv": 3.0, "Dvu": 0.2, "Dvv": 1.0, "a": 0.01, "b": 2.5},
            bc,
            grid,
        )
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "cross-diffusion-schnakenberg", ndim)
        check_dimension_variation(result, ndim, "cross-diffusion-schnakenberg")
