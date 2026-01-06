"""Tests for Nonlinear Beam PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestNonlinearBeamPDE:
    """Tests for Nonlinear Beam PDE."""

    def test_registered(self):
        """Test that nonlinear-beam is registered."""
        assert "nonlinear-beam" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("nonlinear-beam")
        meta = preset.metadata

        assert meta.name == "nonlinear-beam"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test creating the PDE object."""
        preset = get_pde_preset("nonlinear-beam")
        params = preset.get_default_parameters()
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, small_grid)
        # Just verify it creates without error - the nested laplacian
        # is complex and may have stability issues in short tests
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        preset = get_pde_preset("nonlinear-beam")
        state = preset.create_initial_state(small_grid, "default", {"seed": 42})

        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("nonlinear-beam")
        # Use smaller stiffness parameters for stability
        params = {"E_star": 0.0001, "Delta_E": 1.0, "eps": 0.1}
        # Use periodic BC to match the periodic grid
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1, "seed": 42}
        )

        # Run a very short simulation with tiny timestep (fourth-order PDE is stiff)
        result = pde.solve(state, t_range=0.0001, dt=1e-7)

        # Check that result is finite and valid
        assert result is not None
        assert np.isfinite(result.data).all(), "Simulation produced NaN or Inf values"
