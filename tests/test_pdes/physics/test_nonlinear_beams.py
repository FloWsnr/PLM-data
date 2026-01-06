"""Tests for overdamped nonlinear beam with state-dependent stiffness PDE."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestNonlinearBeamsPDE:
    """Tests for overdamped beam with state-dependent stiffness."""

    def test_registered(self):
        """Test that nonlinear-beams is registered."""
        assert "nonlinear-beams" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("nonlinear-beams")
        meta = preset.metadata

        assert meta.name == "nonlinear-beams"
        assert meta.category == "physics"
        assert meta.num_fields == 1  # Now single field (overdamped)
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("nonlinear-beams")
        params = preset.get_default_parameters()

        assert "E_star" in params
        assert "Delta_E" in params
        assert "eps" in params

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("nonlinear-beams")
        # Use smaller Delta_E for stability
        params = {"E_star": 1.0, "Delta_E": 1.0, "eps": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1}
        )

        # Has 4th order terms, needs small dt and implicit solver
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
