"""Tests for cyclic competition (rock-paper-scissors) model."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestCyclicCompetitionPDE:
    """Tests for cyclic competition (rock-paper-scissors) model."""

    def test_registered(self):
        """Test that cyclic-competition is registered."""
        assert "cyclic-competition" in list_presets()

    def test_metadata(self, small_grid):
        """Test metadata."""
        preset = get_pde_preset("cyclic-competition")
        meta = preset.metadata

        assert meta.name == "cyclic-competition"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"u", "v", "w"}

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("cyclic-competition")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
