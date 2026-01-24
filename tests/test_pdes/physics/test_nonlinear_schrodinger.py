"""Tests for Nonlinear Schrodinger PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.nonlinear_schrodinger import NonlinearSchrodingerPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestNonlinearSchrodingerPDE:
    """Tests for Nonlinear Schrodinger PDE."""

    def test_registered(self):
        """Test that nonlinear-schrodinger is registered."""
        assert "nonlinear-schrodinger" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("nonlinear-schrodinger")
        meta = preset.metadata

        assert meta.name == "nonlinear-schrodinger"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("nonlinear-schrodinger", "physics")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "nonlinear-schrodinger"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Nonlinear Schrodinger works in all supported dimensions."""
        np.random.seed(42)
        preset = NonlinearSchrodingerPDE()

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
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "nonlinear-schrodinger", ndim)
        check_dimension_variation(result, ndim, "nonlinear-schrodinger")
