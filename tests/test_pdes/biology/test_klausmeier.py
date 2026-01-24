"""Tests for Klausmeier vegetation-water PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestKlausmeierPDE:
    """Tests for Klausmeier vegetation-water PDE."""

    def test_registered(self):
        """Test that klausmeier is registered."""
        assert "klausmeier" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("klausmeier")
        meta = preset.metadata

        assert meta.name == "klausmeier"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "n" in meta.field_names
        assert "w" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("klausmeier", "biology")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "klausmeier"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Klausmeier works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("klausmeier")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        params = {"a": 0.45, "b": 0.45, "v": 182.5, "D_w": 500.0, "D_n": 1.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "klausmeier", ndim)
        check_dimension_variation(result, ndim, "klausmeier")
