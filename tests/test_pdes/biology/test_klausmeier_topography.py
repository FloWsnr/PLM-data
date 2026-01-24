"""Tests for Klausmeier on topography PDE."""

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

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("klausmeier-topography", "biology")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "klausmeier-topography"

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Klausmeier Topography works in supported dimensions.

        Note: This PDE uses a custom initial condition that assumes 2D+ grids,
        so we only test 2D and 3D here. Uses smaller timestep for numerical stability.
        """
        np.random.seed(42)
        preset = get_pde_preset("klausmeier-topography")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state (use random-uniform for variation testing)
        params = {"a": 0.45, "b": 0.45, "v": 182.5, "D_w": 500.0, "D_n": 1.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation with smaller timestep for stability
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "klausmeier-topography", ndim)
        check_dimension_variation(result, ndim, "klausmeier-topography")
