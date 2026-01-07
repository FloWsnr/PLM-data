"""Tests for Rayleigh-Benard thermal convection PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.core.config import BoundaryConfig, FieldBoundaryConfig


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestThermalConvectionPDE:
    """Tests for Rayleigh-Benard thermal convection."""

    def test_registered(self):
        """Test that thermal-convection is registered."""
        assert "thermal-convection" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("thermal-convection")
        meta = preset.metadata

        assert meta.name == "thermal-convection"
        assert meta.category == "fluids"
        assert meta.num_fields == 3
        assert "omega" in meta.field_names
        assert "psi" in meta.field_names
        assert "b" in meta.field_names

    def test_default_parameters(self):
        """Test default parameters match reference."""
        preset = get_pde_preset("thermal-convection")
        params = preset.get_default_parameters()

        assert "nu" in params
        assert "epsilon" in params
        assert "kappa" in params
        assert "T_b" in params
        assert params["nu"] == 0.2
        assert params["epsilon"] == 0.05
        assert params["kappa"] == 0.5
        assert params["T_b"] == 0.08

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("thermal-convection")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("thermal-convection")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.01, "seed": 42})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()

    def test_per_field_boundary_conditions(self):
        """Test simulation with per-field boundary conditions.

        Uses the physical BCs from Visual PDE reference:
        - omega, psi: Dirichlet=0 top/bottom, periodic left/right
        - b: Dirichlet=0 top, Neumann=T_b bottom (heat flux), periodic left/right
        """
        # Non-periodic grid for Dirichlet BCs on y-axis
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=[True, False])

        preset = get_pde_preset("thermal-convection")
        params = preset.get_default_parameters()

        # Per-field BC configuration
        bc = BoundaryConfig(
            x="periodic",
            y="periodic",  # Default fallback
            fields={
                "omega": FieldBoundaryConfig(
                    x="periodic",
                    top="dirichlet:0",
                    bottom="dirichlet:0"
                ),
                "psi": FieldBoundaryConfig(
                    x="periodic",
                    top="dirichlet:0",
                    bottom="dirichlet:0"
                ),
                "b": FieldBoundaryConfig(
                    x="periodic",
                    top="dirichlet:0",
                    bottom="neumann:0.08"  # T_b heat flux
                ),
            }
        )

        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "default", {"noise": 0.01, "seed": 42})

        # Run a short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()

    def test_per_field_bc_via_dict(self):
        """Test per-field BCs using dict format (as from YAML config)."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=[True, False])

        preset = get_pde_preset("thermal-convection")
        params = preset.get_default_parameters()

        # Dict format (as would come from YAML parsing)
        bc = {
            "x": "periodic",
            "y": "periodic",
            "fields": {
                "omega": {"top": "dirichlet:0", "bottom": "dirichlet:0"},
                "psi": {"top": "dirichlet:0", "bottom": "dirichlet:0"},
                "b": {"top": "dirichlet:0", "bottom": "neumann:0.08"},
            }
        }

        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "default", {"noise": 0.01, "seed": 42})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
