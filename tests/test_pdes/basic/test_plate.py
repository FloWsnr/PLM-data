"""Tests for plate vibration equation PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.plate import PlatePDE


class TestPlatePDE:
    """Tests for the plate vibration equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("plate")
        meta = preset.metadata

        assert meta.name == "plate"
        assert meta.category == "basic"
        assert meta.num_fields == 3  # u, v, w
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names
        # Should mention biharmonic or vibration or wave
        desc_lower = meta.description.lower()
        assert "biharmonic" in desc_lower or "vibration" in desc_lower or "wave" in desc_lower

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("plate")
        params = preset.get_default_parameters()

        assert "D" in params
        assert "Q" in params
        assert "C" in params
        assert "D_c" in params
        assert params["D"] == 10.0
        assert params["Q"] == 0.003
        assert params["C"] == 0.1
        assert params["D_c"] == 0.1

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("plate")
        params = preset.get_default_parameters()
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test initial state creation."""
        preset = get_pde_preset("plate")
        state = preset.create_initial_state(
            small_grid, "constant", {"value": -4.0}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert state[0].label == "u"
        assert state[1].label == "v"
        assert state[2].label == "w"
        # u should be -4, v and w should be zero
        assert np.allclose(state[0].data, -4.0)
        assert np.allclose(state[1].data, 0)
        assert np.allclose(state[2].data, 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "plate" in list_presets()

    def test_short_simulation(self, non_periodic_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("plate")
        params = {"D": 10.0, "Q": 0.003, "C": 0.1, "D_c": 0.1}
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 0.5}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert np.isfinite(result[2].data).all()
