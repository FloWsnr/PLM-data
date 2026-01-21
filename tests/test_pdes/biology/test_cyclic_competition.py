"""Tests for cyclic competition (rock-paper-scissors) model."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


class TestCyclicCompetitionPDE:
    """Tests for cyclic competition (rock-paper-scissors) model."""

    def test_registered(self):
        """Test that cyclic-competition is registered."""
        assert "cyclic-competition" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cyclic-competition")
        meta = preset.metadata

        assert meta.name == "cyclic-competition"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"u", "v", "w"}

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("cyclic-competition", "biology", t_end=0.01)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "cyclic-competition"

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Cyclic Competition works in supported dimensions.

        Note: This PDE uses a custom initial condition that assumes 2D+ grids,
        so we only test 2D and 3D here.
        """
        np.random.seed(42)
        preset = get_pde_preset("cyclic-competition")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state using default IC
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "default", {"noise": 0.01, "seed": 42})

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "cyclic-competition", ndim)
