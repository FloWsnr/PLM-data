"""Tests for Sine-Gordon equation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestSineGordonPDE:
    """Tests for Sine-Gordon equation PDE."""

    def test_registered(self):
        """Test that sine-gordon is registered."""
        assert "sine-gordon" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("sine-gordon")
        meta = preset.metadata

        assert meta.name == "sine-gordon"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "phi" in meta.field_names
        assert "psi" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        from pde import CartesianGrid

        grid = CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)
        preset = get_pde_preset("sine-gordon")
        pde = preset.create_pde(
            {"c": 1.0, "gamma": 0.0},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [2])
    def test_short_simulation(self, ndim: int):
        """Test Sine-Gordon simulation in 2D."""
        np.random.seed(42)
        preset = get_pde_preset("sine-gordon")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        grid = create_grid_for_dimension(ndim, resolution=16)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"c": 1.0, "gamma": 0.0}, bc, grid)
        state = preset.create_initial_state(grid, "random", {"amplitude": 0.5})

        result = pde.solve(state, t_range=0.1, dt=0.01, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "sine-gordon", ndim)
        check_dimension_variation(result, ndim, "sine-gordon")

    def test_unsupported_dimensions(self):
        """Test that sine-gordon only supports 2D."""
        preset = get_pde_preset("sine-gordon")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
