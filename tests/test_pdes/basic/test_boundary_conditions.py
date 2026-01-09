"""Tests for boundary condition conversion."""

import numpy as np
import pytest

from pde_sim.pdes import get_pde_preset
from pde_sim.core.config import BoundaryConfig


class TestBoundaryConditions:
    """Tests for boundary condition conversion."""

    def test_convert_bc_periodic(self):
        """Test periodic BC conversion."""
        preset = get_pde_preset("heat")
        bc = BoundaryConfig()  # Default is all periodic
        result = preset._convert_bc(bc)
        assert result == {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"}

    def test_convert_bc_no_flux(self):
        """Test no-flux (Neumann) BC conversion."""
        preset = get_pde_preset("heat")
        bc = BoundaryConfig(
            x_minus="neumann:0", x_plus="neumann:0",
            y_minus="neumann:0", y_plus="neumann:0"
        )
        result = preset._convert_bc(bc)
        assert result == {
            "x-": {"derivative": 0},
            "x+": {"derivative": 0},
            "y-": {"derivative": 0},
            "y+": {"derivative": 0},
        }

    def test_convert_bc_dirichlet(self):
        """Test Dirichlet BC conversion."""
        preset = get_pde_preset("heat")
        bc = BoundaryConfig(
            x_minus="dirichlet:0", x_plus="dirichlet:0",
            y_minus="dirichlet:0", y_plus="dirichlet:0"
        )
        result = preset._convert_bc(bc)
        assert result == {
            "x-": {"value": 0},
            "x+": {"value": 0},
            "y-": {"value": 0},
            "y+": {"value": 0},
        }

    def test_convert_bc_mixed(self):
        """Test mixed BC conversion."""
        preset = get_pde_preset("heat")
        bc = BoundaryConfig(
            x_minus="periodic", x_plus="periodic",
            y_minus="neumann:0", y_plus="neumann:0"
        )
        result = preset._convert_bc(bc)
        assert result == {
            "x-": "periodic",
            "x+": "periodic",
            "y-": {"derivative": 0},
            "y+": {"derivative": 0},
        }

    def test_simulation_with_no_flux_bc(self, non_periodic_grid):
        """Test running simulation with no-flux boundary conditions."""
        preset = get_pde_preset("heat")
        params = {"D_T": 0.01}
        bc = BoundaryConfig(
            x_minus="neumann:0", x_plus="neumann:0",
            y_minus="neumann:0", y_plus="neumann:0"
        )

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert np.all(np.isfinite(result.data))
