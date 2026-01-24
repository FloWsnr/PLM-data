"""Tests for standing wave equation PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.wave_standing import StandingWavePDE

from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestStandingWavePDE:
    """Tests for the standing wave equation preset."""

    def test_registered(self):
        """Test that wave-standing is registered."""
        assert "wave-standing" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("wave-standing")
        meta = preset.metadata

        assert meta.name == "wave-standing"
        assert meta.category == "basic"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        preset = get_pde_preset("wave-standing")
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)
        params = {"D": 1.0}

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None

    def test_create_initial_state(self):
        """Test initial state creation."""
        preset = get_pde_preset("wave-standing")
        grid = create_grid_for_dimension(2, resolution=16)

        state = preset.create_initial_state(
            grid, "gaussian-blobs", {"num_blobs": 1, "seed": 42}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"
        # Velocity should start at zero
        assert np.allclose(state[1].data, 0)

    def test_short_simulation(self):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = StandingWavePDE()
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        params = {"D": 1.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "gaussian-blobs", {"num_blobs": 1})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test standing wave equation works in all supported dimensions."""
        np.random.seed(42)
        preset = StandingWavePDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        params = {"D": 1.0}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "wave-standing", ndim)
        check_dimension_variation(result, ndim, "wave-standing")
