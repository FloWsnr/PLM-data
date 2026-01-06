"""Tests for boundary condition conversion."""

import numpy as np
import pytest

from pde_sim.pdes import get_pde_preset


class TestBoundaryConditions:
    """Tests for boundary condition conversion."""

    def test_convert_bc_periodic(self):
        """Test periodic BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "periodic", "y": "periodic"})
        assert bc == ["periodic", "periodic"]

    def test_convert_bc_no_flux(self):
        """Test no-flux (Neumann) BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "no-flux", "y": "no-flux"})
        assert bc == [{"derivative": 0}, {"derivative": 0}]

    def test_convert_bc_dirichlet(self):
        """Test Dirichlet BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "dirichlet", "y": "dirichlet"})
        assert bc == [{"value": 0}, {"value": 0}]

    def test_convert_bc_mixed(self):
        """Test mixed BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "periodic", "y": "no-flux"})
        assert bc == ["periodic", {"derivative": 0}]

    def test_simulation_with_no_flux_bc(self, non_periodic_grid):
        """Test running simulation with no-flux boundary conditions."""
        preset = get_pde_preset("heat")
        params = {"D_T": 0.01}
        bc = {"x": "no-flux", "y": "no-flux"}

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert np.all(np.isfinite(result.data))
