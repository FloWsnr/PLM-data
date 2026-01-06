"""Tests for Swift-Hohenberg with Advection PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestSwiftHohenbergAdvectionPDE:
    """Tests for Swift-Hohenberg with Advection PDE."""

    def test_registered(self):
        """Test that swift-hohenberg-advection is registered."""
        assert "swift-hohenberg-advection" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("swift-hohenberg-advection")
        meta = preset.metadata

        assert meta.name == "swift-hohenberg-advection"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("swift-hohenberg-advection")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"seed": 42})

        result = pde.solve(state, t_range=0.01, dt=0.0008, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
