"""Tests for Schrodinger equation PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.schrodinger import SchrodingerPDE


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
        bc = {"x": "dirichlet", "y": "dirichlet"}

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

    def test_short_simulation(self, non_periodic_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("schrodinger")
        params = {"D": 1.0, "C": 0.004}
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "eigenstate", {"n": 3, "m": 3}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
