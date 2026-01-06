"""Tests for potential flow dipoles PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)


class TestPotentialFlowDipolesPDE:
    """Tests for potential flow dipoles."""

    def test_registered(self):
        """Test that potential-flow-dipoles is registered."""
        assert "potential-flow-dipoles" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("potential-flow-dipoles")
        meta = preset.metadata

        assert meta.name == "potential-flow-dipoles"
        assert meta.category == "fluids"
        assert meta.num_fields == 2
        assert "phi" in meta.field_names
        assert "s" in meta.field_names

    def test_default_parameters(self):
        """Test default parameters match reference."""
        preset = get_pde_preset("potential-flow-dipoles")
        params = preset.get_default_parameters()

        assert "d" in params
        assert "strength" in params
        assert params["d"] == 5.0
        assert params["strength"] == 1000.0

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("potential-flow-dipoles")
        params = preset.get_default_parameters()
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test dipole initial condition."""
        preset = get_pde_preset("potential-flow-dipoles")
        state = preset.create_initial_state(
            small_grid, "dipole", {"d": 5.0, "strength": 1000.0}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        phi = state[0]
        # Potential should have variation (source-sink pair)
        assert np.std(phi.data) > 0
        assert np.isfinite(phi.data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("potential-flow-dipoles")
        params = preset.get_default_parameters()
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "dipole", {"d": 5.0})

        result = pde.solve(state, t_range=0.1, dt=0.01, solver="euler")

        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
