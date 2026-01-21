"""Tests for Keller-Segel PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestKellerSegelPDE:
    """Tests for the Keller-Segel chemotaxis model."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("keller-segel")
        meta = preset.metadata

        assert meta.name == "keller-segel"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names  # cells
        assert "v" in meta.field_names  # chemoattractant

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("keller-segel")
        params = preset.get_default_parameters()

        assert "c" in params  # chemotaxis coefficient
        assert "D" in params  # diffusion ratio
        assert "a" in params  # decay rate

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("keller-segel")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_default(self, small_grid):
        """Test default initial state creation."""
        preset = get_pde_preset("keller-segel")
        state = preset.create_initial_state(
            small_grid, "keller-segel-default", {}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "keller-segel" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("keller-segel", "biology", t_end=0.01)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "keller-segel"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Keller-Segel works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("keller-segel")

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
        check_result_finite(result, "keller-segel", ndim)
