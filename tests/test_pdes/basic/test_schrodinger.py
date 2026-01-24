"""Tests for Schrodinger equation PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.core.config import BoundaryConfig
from pde_sim.pdes.basic.schrodinger import SchrodingerPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


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

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("schrodinger")
        params = {"D": 1.0, "C": 1.0, "n": 1, "m": 1, "V_strength": 0.0, "pot_n": 2, "pot_m": 2}
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
        result, config = run_short_simulation("schrodinger", "basic")

        assert result is not None
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "schrodinger"

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

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", backend="numpy")

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

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Schrodinger equation works in all supported dimensions."""
        np.random.seed(42)
        preset = SchrodingerPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs - use non-periodic for Schrodinger (box)
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=False)
        bc = create_bc_for_dimension(ndim, periodic=False)

        # Create PDE and initial state using random-uniform (works in all dimensions)
        params = {"D": 1.0, "C": 1.0, "n": 1, "m": 1, "V_strength": 0.0, "pot_n": 2, "pot_m": 2}
        pde = preset.create_pde(params, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "schrodinger", ndim)
        check_dimension_variation(result, ndim, "schrodinger")
