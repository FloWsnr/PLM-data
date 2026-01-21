"""Tests for Van der Pol Oscillator PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.van_der_pol import VanDerPolPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestVanDerPolPDE:
    """Tests for Van der Pol Oscillator PDE."""

    def test_registered(self):
        """Test that van-der-pol is registered."""
        assert "van-der-pol" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("van-der-pol")
        meta = preset.metadata

        assert meta.name == "van-der-pol"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "X" in meta.field_names
        assert "Y" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("van-der-pol", "physics", t_end=0.01)

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "van-der-pol"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Van der Pol works in all supported dimensions."""
        np.random.seed(42)
        preset = VanDerPolPDE()

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
        check_result_finite(result, "van-der-pol", ndim)
