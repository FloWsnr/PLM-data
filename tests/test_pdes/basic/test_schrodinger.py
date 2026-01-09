"""Tests for Schrodinger equation PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.core.config import BoundaryConfig
from pde_sim.pdes.basic.schrodinger import SchrodingerPDE

from tests.conftest import run_short_simulation


class TestSchrodingerPDE:
    """Tests for the Schrodinger equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("schrodinger")
        meta = preset.metadata

        assert meta.name == "schrodinger"
        assert meta.category == "basic"
        assert meta.num_fields == 2  # Now uses real u, v components
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("schrodinger")
        params = preset.get_default_parameters()

        assert "D" in params
        assert "C" in params
        assert "n" in params
        assert "m" in params
        assert params["D"] == 1.0
        assert params["C"] == 0.004
        assert params["n"] == 3
        assert params["m"] == 3

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("schrodinger")
        params = preset.get_default_parameters()
        bc = BoundaryConfig(
            x_minus="dirichlet:0", x_plus="dirichlet:0",
            y_minus="dirichlet:0", y_plus="dirichlet:0"
        )

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_eigenstate(self, small_grid):
        """Test eigenstate initial condition."""
        preset = get_pde_preset("schrodinger")
        state = preset.create_initial_state(
            small_grid, "eigenstate", {"n": 3, "m": 3}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        # Should have non-zero u (real part)
        assert np.any(state[0].data != 0)
        # v (imaginary part) should start at zero for eigenstate
        assert np.allclose(state[1].data, 0)

    def test_create_initial_state_wave_packet(self, small_grid):
        """Test wave packet initial condition."""
        preset = get_pde_preset("schrodinger")
        state = preset.create_initial_state(
            small_grid, "wave-packet", {"kx": 5.0, "sigma": 0.1}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        # Should have non-zero data
        norm = np.sum(state[0].data**2 + state[1].data**2)
        assert norm > 0

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "schrodinger" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("schrodinger", "basic", t_end=0.001)

        assert result is not None
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "schrodinger"

    def test_potential_parameters_in_metadata(self):
        """Test that potential parameters are in metadata."""
        preset = get_pde_preset("schrodinger")
        params = preset.get_default_parameters()

        assert "V_strength" in params
        assert "pot_n" in params
        assert "pot_m" in params
        assert params["V_strength"] == 0.0  # Default: no potential

    def test_create_pde_with_sinusoidal_potential(self, non_periodic_grid):
        """Test PDE creation with sinusoidal potential."""
        preset = get_pde_preset("schrodinger")
        params = {
            "D": 1.0,
            "C": 0.004,
            "potential_type": "sinusoidal",
            "V_strength": 1.0,
            "pot_n": 5,
            "pot_m": 5,
        }
        bc = BoundaryConfig(
            x_minus="dirichlet:0", x_plus="dirichlet:0",
            y_minus="dirichlet:0", y_plus="dirichlet:0"
        )

        pde = preset.create_pde(params, bc, non_periodic_grid)

        assert pde is not None
        # Verify consts contains V field
        assert pde.consts is not None
        assert "V" in pde.consts

    def test_create_pde_with_harmonic_potential(self, non_periodic_grid):
        """Test PDE creation with harmonic potential."""
        preset = get_pde_preset("schrodinger")
        params = {
            "D": 1.0,
            "C": 0.004,
            "potential_type": "harmonic",
            "V_strength": 0.1,
        }
        bc = BoundaryConfig(
            x_minus="dirichlet:0", x_plus="dirichlet:0",
            y_minus="dirichlet:0", y_plus="dirichlet:0"
        )

        pde = preset.create_pde(params, bc, non_periodic_grid)

        assert pde is not None
        assert pde.consts is not None
        assert "V" in pde.consts
        # Harmonic potential should be positive everywhere except center
        V_field = pde.consts["V"]
        assert np.all(V_field.data >= 0)

    def test_create_pde_no_potential(self, non_periodic_grid):
        """Test PDE creation without potential (default behavior)."""
        preset = get_pde_preset("schrodinger")
        params = {"D": 1.0, "C": 0.004}
        bc = BoundaryConfig(
            x_minus="dirichlet:0", x_plus="dirichlet:0",
            y_minus="dirichlet:0", y_plus="dirichlet:0"
        )

        pde = preset.create_pde(params, bc, non_periodic_grid)

        assert pde is not None
        # No consts when potential is not used
        assert pde.consts is None or "V" not in (pde.consts or {})

    def test_short_simulation_with_potential(self, non_periodic_grid):
        """Test running simulation with sinusoidal potential."""
        preset = get_pde_preset("schrodinger")
        params = {
            "D": 1.0,
            "C": 0.004,
            "potential_type": "sinusoidal",
            "V_strength": 0.5,
            "pot_n": 3,
            "pot_m": 3,
        }
        bc = BoundaryConfig(
            x_minus="dirichlet:0", x_plus="dirichlet:0",
            y_minus="dirichlet:0", y_plus="dirichlet:0"
        )

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "eigenstate", {"n": 2, "m": 2}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()

    def test_localized_initial_condition(self, non_periodic_grid):
        """Test localized initial condition for potential simulations."""
        preset = get_pde_preset("schrodinger")
        state = preset.create_initial_state(
            non_periodic_grid, "localized", {"power": 10}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        # Should have non-zero u (real part)
        assert np.any(state[0].data != 0)
        # v (imaginary part) should start at zero
        assert np.allclose(state[1].data, 0)

    def test_get_equations_for_metadata_with_potential(self):
        """Test equations include potential when present."""
        preset = get_pde_preset("schrodinger")
        params = {
            "D": 1.0,
            "C": 0.004,
            "potential_type": "sinusoidal",
            "V_strength": 2.0,
        }

        equations = preset.get_equations_for_metadata(params)

        assert "V*v" in equations["u"]
        assert "V*u" in equations["v"]
        assert "V(x,y)" in equations

    def test_get_equations_for_metadata_no_potential(self):
        """Test equations without potential."""
        preset = get_pde_preset("schrodinger")
        params = {"D": 1.0, "C": 0.004}

        equations = preset.get_equations_for_metadata(params)

        assert "V" not in equations["u"]
        assert "V" not in equations["v"]
        assert "V(x,y)" not in equations
