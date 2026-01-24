"""Tests for plate vibration equation PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.plate import PlatePDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestPlatePDE:
    """Tests for the plate vibration equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("plate")
        meta = preset.metadata

        assert meta.name == "plate"
        assert meta.category == "basic"
        assert meta.num_fields == 3  # u, v, w
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names
        # Should mention biharmonic or vibration or wave
        desc_lower = meta.description.lower()
        assert "biharmonic" in desc_lower or "vibration" in desc_lower or "wave" in desc_lower

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("plate")
        params = {"D": 1.0, "Q": 10.0, "C": 0.1, "D_c": 0.1}
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test initial state creation."""
        preset = get_pde_preset("plate")
        state = preset.create_initial_state(
            small_grid, "constant", {"value": -4.0}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert state[0].label == "u"
        assert state[1].label == "v"
        assert state[2].label == "w"
        # u should be -4, v and w should be zero
        assert np.allclose(state[0].data, -4.0)
        assert np.allclose(state[1].data, 0)
        assert np.allclose(state[2].data, 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "plate" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("plate", "basic")

        assert result is not None
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert np.isfinite(result[2].data).all()
        assert config["preset"] == "plate"

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_dimension_support(self, ndim: int):
        """Test plate equation works in supported dimensions (1D and 2D)."""
        np.random.seed(42)
        preset = PlatePDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs - use non-periodic for plate equation
        resolution = 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        # Create PDE and initial state (use random-uniform for variation testing)
        params = {"D": 1.0, "Q": 10.0, "C": 0.1, "D_c": 0.1}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": -5.0, "high": -3.0})

        # Run short simulation
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "plate", ndim)
        check_dimension_variation(result, ndim, "plate")

    def test_dimension_3d_not_supported(self):
        """Test that plate rejects 3D (BC conversion issue)."""
        preset = PlatePDE()
        assert 3 not in preset.metadata.supported_dimensions
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
