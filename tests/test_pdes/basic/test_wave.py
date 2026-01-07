"""Tests for wave equation PDEs."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.wave import WavePDE, InhomogeneousWavePDE


class TestWavePDE:
    """Tests for the Wave equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("wave")
        meta = preset.metadata

        assert meta.name == "wave"
        assert meta.category == "basic"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("wave")
        params = preset.get_default_parameters()

        assert "D" in params
        assert "C" in params
        assert params["D"] == 1.0
        assert params["C"] == 0.01

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("wave")
        params = {"D": 1.0, "C": 0.01}
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test initial state creation."""
        preset = get_pde_preset("wave")
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"
        # Velocity should start at zero
        assert np.allclose(state[1].data, 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "wave" in list_presets()

    def test_short_simulation(self, non_periodic_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("wave")
        params = {"D": 1.0, "C": 0.01}
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 0.5}
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()


class TestInhomogeneousWavePDE:
    """Tests for the inhomogeneous wave equation with spatially varying wave speed."""

    def test_registered(self):
        """Test that inhomogeneous-wave is registered."""
        assert "inhomogeneous-wave" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("inhomogeneous-wave")
        meta = preset.metadata

        assert meta.name == "inhomogeneous-wave"
        assert meta.category == "basic"
        assert meta.num_fields == 2
        assert "u" in meta.field_names  # displacement
        assert "v" in meta.field_names  # velocity

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("inhomogeneous-wave")
        params = preset.get_default_parameters()

        assert "D" in params  # base diffusivity (wave speed squared)
        assert "E" in params  # amplitude of spatial variation
        assert "m" in params  # spatial mode x
        assert "n" in params  # spatial mode y
        assert "C" in params  # damping
        assert params["D"] == 1.0
        assert params["E"] == 0.97
        assert params["m"] == 9
        assert params["n"] == 9
        assert params["C"] == 0.01

    def test_create_pde(self, non_periodic_grid):
        """Test PDE creation."""
        preset = get_pde_preset("inhomogeneous-wave")
        params = preset.get_default_parameters()
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, non_periodic_grid)
        assert pde is not None

    def test_create_initial_state(self, non_periodic_grid):
        """Test initial state creation."""
        preset = get_pde_preset("inhomogeneous-wave")
        state = preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"
        # Velocity should start at zero
        assert np.allclose(state[1].data, 0)

    def test_short_simulation(self, non_periodic_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("inhomogeneous-wave")
        params = {"D": 1.0, "E": 0.5, "m": 2, "n": 2, "C": 0.01}
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1}
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
