"""Tests for Lotka-Volterra predator-prey PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.initial_conditions import create_initial_condition
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


def _create_multifield_initial_state(grid, preset, ic_params):
    """Create FieldCollection initial state that works for any dimension.

    This helper works around PDE presets whose create_initial_state only
    supports 2D grids.
    """
    fields = []
    for name in preset.metadata.field_names:
        ic = create_initial_condition(grid, "random-uniform", ic_params)
        ic.label = name
        fields.append(ic)
    return FieldCollection(fields)


class TestLotkaVolterraPDE:
    """Tests for Lotka-Volterra predator-prey model."""

    def test_registered(self):
        """Test that lotka-volterra is registered."""
        assert "lotka-volterra" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("lotka-volterra")
        meta = preset.metadata

        assert meta.name == "lotka-volterra"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert set(meta.field_names) == {"u", "v"}

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("lotka-volterra")
        pde = preset.create_pde(
            {"Du": 1.0, "Dv": 0.5, "alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 0.5},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Lotka-Volterra works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("lotka-volterra")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde(
            {"Du": 1.0, "Dv": 0.5, "alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 0.5},
            bc,
            grid,
        )

        # Use helper to create dimension-agnostic initial state
        # (the preset's create_initial_state has a bug for 1D/3D grids)
        state = _create_multifield_initial_state(grid, preset, {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "lotka-volterra", ndim)
        check_dimension_variation(result, ndim, "lotka-volterra")
