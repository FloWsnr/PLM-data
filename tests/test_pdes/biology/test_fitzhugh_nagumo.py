"""Tests for FitzHugh-Nagumo PDE."""

import numpy as np
import pytest
from pde import FieldCollection

from pde_sim.pdes import get_pde_preset
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestFitzHughNagumoPDE:
    """Tests for FitzHugh-Nagumo PDE."""

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fitzhugh-nagumo")
        meta = preset.metadata

        assert meta.name == "fitzhugh-nagumo"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test FitzHugh-Nagumo works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("fitzhugh-nagumo")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"Dv": 20.0, "e_v": 0.5, "a_v": 1.0, "a_z": -0.1}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "fitzhugh-nagumo", ndim)
        check_dimension_variation(result, ndim, "fitzhugh-nagumo")

    def test_spiral_seed_default_center(self):
        """Test spiral-seed IC defaults to domain center."""
        preset = get_pde_preset("fitzhugh-nagumo")
        grid = create_grid_for_dimension(2, resolution=32)

        ic_params = {
            "u_rest": -0.66, "v_rest": -0.37,
            "u_excited": 1.0, "v_refractory": 0.5,
        }
        resolved = preset.resolve_ic_params(grid, "spiral-seed", ic_params)
        assert resolved["x_center"] == pytest.approx(0.5)  # midpoint of [0, 1]
        assert resolved["y_center"] == pytest.approx(0.5)

        state = preset.create_initial_state(grid, "spiral-seed", resolved)
        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.all(np.isfinite(state[0].data))
        assert np.all(np.isfinite(state[1].data))

    def test_spiral_seed_random_position(self):
        """Test spiral-seed IC with randomized step positions."""
        preset = get_pde_preset("fitzhugh-nagumo")
        grid = create_grid_for_dimension(2, resolution=32)

        ic_params = {
            "u_rest": -0.66, "v_rest": -0.37,
            "u_excited": 1.0, "v_refractory": 0.5,
            "x_center": "random", "y_center": "random",
        }
        resolved = preset.resolve_ic_params(grid, "spiral-seed", ic_params)

        # Should be within 20%-80% of domain [0, 1]
        assert 0.2 <= resolved["x_center"] <= 0.8
        assert 0.2 <= resolved["y_center"] <= 0.8
        # Should no longer be string
        assert isinstance(resolved["x_center"], float)
        assert isinstance(resolved["y_center"], float)

        state = preset.create_initial_state(grid, "spiral-seed", resolved)
        assert isinstance(state, FieldCollection)
        assert np.all(np.isfinite(state[0].data))

    def test_spiral_seed_explicit_position(self):
        """Test spiral-seed IC with explicit step positions."""
        preset = get_pde_preset("fitzhugh-nagumo")
        grid = create_grid_for_dimension(2, resolution=32)

        ic_params = {
            "u_rest": -0.66, "v_rest": -0.37,
            "u_excited": 1.0, "v_refractory": 0.5,
            "x_center": 3.0, "y_center": 7.0,
        }
        resolved = preset.resolve_ic_params(grid, "spiral-seed", ic_params)
        assert resolved["x_center"] == 3.0
        assert resolved["y_center"] == 7.0

        state = preset.create_initial_state(grid, "spiral-seed", resolved)
        assert isinstance(state, FieldCollection)
        assert np.all(np.isfinite(state[0].data))

    def test_spiral_seed_1d(self):
        """Test spiral-seed IC in 1D (only x_center, no y_center)."""
        preset = get_pde_preset("fitzhugh-nagumo")
        grid = create_grid_for_dimension(1, resolution=64)

        ic_params = {
            "u_rest": -0.66, "v_rest": -0.37,
            "u_excited": 1.0, "v_refractory": 0.5,
            "x_center": "random",
        }
        resolved = preset.resolve_ic_params(grid, "spiral-seed", ic_params)
        assert 0.2 <= resolved["x_center"] <= 0.8
        assert "y_center" not in resolved  # 1D has no y

        state = preset.create_initial_state(grid, "spiral-seed", resolved)
        assert isinstance(state, FieldCollection)
        assert len(state) == 2

    def test_spiral_seed_position_params(self):
        """Test get_position_params returns correct params for spiral-seed."""
        preset = get_pde_preset("fitzhugh-nagumo")
        assert preset.get_position_params("spiral-seed") == {"x_center", "y_center"}
