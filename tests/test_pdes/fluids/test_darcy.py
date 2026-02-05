"""Tests for Darcy Flow PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestDarcyPDE:
    """Tests for the Darcy Flow PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("darcy")
        meta = preset.metadata

        assert meta.name == "darcy"
        assert meta.category == "fluids"
        assert meta.num_fields == 1
        assert "p" in meta.field_names
        assert meta.supported_dimensions == [1, 2, 3]
        assert len(meta.parameters) == 2
        param_names = [p.name for p in meta.parameters]
        assert "K" in param_names
        assert "f" in param_names

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test running a short simulation in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("darcy")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        params = {"K": 1.0, "f": 0.0}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "darcy", ndim)
        check_dimension_variation(result, ndim, "darcy")

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_simulation_with_source(self, ndim: int):
        """Test simulation with non-zero source term in all dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("darcy")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        params = {"K": 1.0, "f": 0.1}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        check_result_finite(result, "darcy", ndim)
        check_dimension_variation(result, ndim, "darcy")

    def test_get_equations_for_metadata(self):
        """Test that equation strings are generated correctly."""
        preset = get_pde_preset("darcy")
        params = {"K": 2.5, "f": 0.3}

        equations = preset.get_equations_for_metadata(params)

        assert "p" in equations
        assert "2.5" in equations["p"]
        assert "0.3" in equations["p"]
        assert "laplace(p)" in equations["p"]

    def test_missing_parameters_raises_error(self):
        """Test that missing parameters raise KeyError."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("darcy")
        bc = {"x": "periodic", "y": "periodic"}

        with pytest.raises(KeyError):
            preset.create_pde({}, bc, grid)

        with pytest.raises(KeyError):
            preset.create_pde({"K": 1.0}, bc, grid)

        with pytest.raises(KeyError):
            preset.create_pde({"f": 0.0}, bc, grid)
