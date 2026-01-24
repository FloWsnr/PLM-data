"""Tests for Gierer-Meinhardt PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
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


class TestGiererMeinhardtPDE:
    """Tests for the Gierer-Meinhardt activator-inhibitor system."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("gierer-meinhardt")
        meta = preset.metadata

        assert meta.name == "gierer-meinhardt"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("gierer-meinhardt")
        params = {"rho": 0.001, "mu": 0.02, "D_a": 0.1, "D_h": 10.0, "kappa": 0.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test initial state creation."""
        preset = get_pde_preset("gierer-meinhardt")
        state = preset.create_initial_state(
            small_grid, "default", {"noise": 0.01, "seed": 42}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"
        # All values should be positive
        assert np.all(state[0].data > 0)
        assert np.all(state[1].data > 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "gierer-meinhardt" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("gierer-meinhardt", "biology")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "gierer-meinhardt"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Gierer-Meinhardt works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("gierer-meinhardt")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        params = {"rho": 0.001, "mu": 0.02, "D_a": 0.1, "D_h": 10.0, "kappa": 0.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "gierer-meinhardt", ndim)
        check_dimension_variation(result, ndim, "gierer-meinhardt")
