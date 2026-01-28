"""Tests for cyclic competition (rock-paper-scissors) model."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

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


class TestCyclicCompetitionPDE:
    """Tests for cyclic competition (rock-paper-scissors) model."""

    def test_registered(self):
        """Test that cyclic-competition is registered."""
        assert "cyclic-competition" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cyclic-competition")
        meta = preset.metadata

        assert meta.name == "cyclic-competition"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"u", "v", "w"}

    def test_create_pde(self):
        """Test PDE creation."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)
        preset = get_pde_preset("cyclic-competition")
        pde = preset.create_pde(
            {"a": 0.8, "b": 1.9, "Du": 2.0, "Dv": 0.5, "Dw": 0.5},
            {"x": "periodic", "y": "periodic"},
            grid,
        )
        assert pde is not None

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_short_simulation(self, ndim: int):
        """Test Cyclic Competition works in all supported dimensions."""
        np.random.seed(42)
        preset = get_pde_preset("cyclic-competition")

        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        pde = preset.create_pde({"a": 0.8, "b": 1.9, "Du": 2.0, "Dv": 0.5, "Dw": 0.5}, bc, grid)

        # Use helper to create dimension-agnostic initial state
        # (the preset's create_initial_state has a bug for 1D/3D grids)
        state = _create_multifield_initial_state(grid, preset, {"low": 0.1, "high": 0.9})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "cyclic-competition", ndim)
        check_dimension_variation(result, ndim, "cyclic-competition")
