"""Tests for Cahn-Hilliard phase separation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestCahnHilliardPDE:
    """Tests for Cahn-Hilliard phase separation."""

    def test_registered(self):
        """Test that cahn-hilliard is registered."""
        assert "cahn-hilliard" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cahn-hilliard")
        meta = preset.metadata

        assert meta.name == "cahn-hilliard"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("cahn-hilliard")
        pde = preset.create_pde(
            {"D": 1.0, "gamma": 1.0, "epsilon": 0.1},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Cahn-Hilliard works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("cahn-hilliard")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"D": 1.0, "gamma": 1.0, "epsilon": 0.1}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": -0.1, "high": 0.1})

        # Run short simulation (4th order PDE is stiff)
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "cahn-hilliard", ndim)
        check_dimension_variation(result, ndim, "cahn-hilliard")
