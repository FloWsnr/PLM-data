"""Tests for Rayleigh-Benard thermal convection PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.core.config import BoundaryConfig
from tests.conftest import run_short_simulation


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

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("thermal-convection")
        params = {"nu": 0.01, "epsilon": 0.1, "kappa": 0.01, "T_b": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("thermal-convection", "fluids")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
        assert config["preset"] == "thermal-convection"

    def test_per_field_boundary_conditions(self):
        """Test simulation with per-field boundary conditions.

        Uses the physical BCs from Visual PDE reference:
        - omega, psi: Dirichlet=0 top/bottom, periodic left/right
        - b: Dirichlet=0 top, Neumann=T_b bottom (heat flux), periodic left/right
        """
        # Non-periodic grid for Dirichlet BCs on y-axis
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=[True, False])

        preset = get_pde_preset("thermal-convection")
        params = {"nu": 0.01, "epsilon": 0.1, "kappa": 0.01, "T_b": 1.0}

        # Per-field BC configuration
        bc = BoundaryConfig(
            x_minus="periodic",
            x_plus="periodic",
            y_minus="periodic",
            y_plus="periodic",
            fields={
                "omega": {"y-": "dirichlet:0", "y+": "dirichlet:0"},
                "psi": {"y-": "dirichlet:0", "y+": "dirichlet:0"},
                "b": {"y-": "neumann:0.08", "y+": "dirichlet:0"},  # T_b heat flux at bottom
            }
        )

        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "default", {"noise": 0.01, "seed": 42})

        # Run a short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", backend="numpy")

        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()

    def test_per_field_bc_via_dict(self):
        """Test per-field BCs using dict format (as from YAML config)."""
        grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=[True, False])

        preset = get_pde_preset("thermal-convection")
        params = {"nu": 0.01, "epsilon": 0.1, "kappa": 0.01, "T_b": 1.0}

        # Dict format (as would come from YAML parsing)
        bc = {
            "x-": "periodic",
            "x+": "periodic",
            "y-": "periodic",
            "y+": "periodic",
            "fields": {
                "omega": {"y-": "dirichlet:0", "y+": "dirichlet:0"},
                "psi": {"y-": "dirichlet:0", "y+": "dirichlet:0"},
                "b": {"y-": "neumann:0.08", "y+": "dirichlet:0"},
            }
        }

        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "default", {"noise": 0.01, "seed": 42})

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", backend="numpy")

        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()

    def test_dimension_support_2d_only(self):
        """Test that thermal-convection only supports 2D."""
        preset = get_pde_preset("thermal-convection")

        # Verify only 2D is supported
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
