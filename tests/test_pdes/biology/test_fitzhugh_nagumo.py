"""Tests for FitzHugh-Nagumo PDE."""

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


class TestFitzHughNagumoPDE:
    """Tests for FitzHugh-Nagumo PDE."""

    def test_registered(self):
        """Test that fitzhugh-nagumo is registered."""
        assert "fitzhugh-nagumo" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fitzhugh-nagumo")
        meta = preset.metadata

        assert meta.name == "fitzhugh-nagumo"
        assert meta.category == "biology"

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("fitzhugh-nagumo", "biology", t_end=0.01)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "fitzhugh-nagumo"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test FitzHugh-Nagumo works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("fitzhugh-nagumo")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "fitzhugh-nagumo", ndim)
        check_dimension_variation(result, ndim, "fitzhugh-nagumo")
