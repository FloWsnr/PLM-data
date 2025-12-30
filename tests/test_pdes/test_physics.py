"""Tests for physics PDEs."""

import numpy as np
import pytest
from pde import PDE, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.gray_scott import GrayScottPDE


class TestGrayScottPDE:
    """Tests for the Gray-Scott reaction-diffusion preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        pde = GrayScottPDE()
        meta = pde.metadata

        assert meta.name == "gray-scott"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert len(meta.parameters) == 4

    def test_get_default_parameters(self):
        """Test getting default parameters."""
        pde = GrayScottPDE()
        defaults = pde.get_default_parameters()

        assert "F" in defaults
        assert "k" in defaults
        assert "Du" in defaults
        assert "Dv" in defaults

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid params."""
        pde = GrayScottPDE()
        # Should not raise
        pde.validate_parameters({"F": 0.04, "k": 0.06})

    def test_validate_parameters_invalid(self):
        """Test parameter validation with invalid params."""
        pde = GrayScottPDE()
        with pytest.raises(ValueError, match="F must be <="):
            pde.validate_parameters({"F": 0.5})  # F max is 0.1

    def test_create_pde(self, small_grid):
        """Test creating the PDE object."""
        pde_preset = GrayScottPDE()
        pde = pde_preset.create_pde(
            parameters={"F": 0.04, "k": 0.06, "Du": 0.16, "Dv": 0.08},
            bc={"x": "periodic", "y": "periodic"},
            grid=small_grid,
        )

        assert isinstance(pde, PDE)

    def test_create_initial_state_default(self, small_grid):
        """Test creating initial state with Gaussian blobs."""
        np.random.seed(42)
        pde = GrayScottPDE()
        state = pde.create_initial_state(
            grid=small_grid,
            ic_type="gaussian-blobs",
            ic_params={"num_blobs": 3},
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        # Check field labels
        assert state[0].label == "u"
        assert state[1].label == "v"

    def test_create_initial_state_gray_scott_default(self, small_grid):
        """Test creating initial state with Gray-Scott specific init."""
        np.random.seed(42)
        pde = GrayScottPDE()
        state = pde.create_initial_state(
            grid=small_grid,
            ic_type="gray-scott-default",
            ic_params={"perturbation_radius": 0.1},
        )

        assert isinstance(state, FieldCollection)
        # u should start mostly at 1.0
        assert np.mean(state[0].data) > 0.8
        # v should start mostly at 0.0
        assert np.mean(state[1].data) < 0.2

    def test_registered_in_registry(self):
        """Test that gray-scott PDE is registered."""
        presets = list_presets()
        assert "gray-scott" in presets

        # Can retrieve via registry
        pde = get_pde_preset("gray-scott")
        assert isinstance(pde, GrayScottPDE)

    def test_short_simulation(self, small_grid):
        """Test running a short simulation with Gray-Scott."""
        np.random.seed(42)
        pde_preset = GrayScottPDE()

        params = {"F": 0.04, "k": 0.06, "Du": 0.16, "Dv": 0.08}
        pde = pde_preset.create_pde(
            parameters=params,
            bc={"x": "periodic", "y": "periodic"},
            grid=small_grid,
        )

        state = pde_preset.create_initial_state(
            grid=small_grid,
            ic_type="gray-scott-default",
            ic_params={},
        )

        # Run short simulation
        result = pde.solve(state, t_range=1.0, dt=0.5, tracker=None)

        # Result should be a FieldCollection
        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        # Values should be finite
        assert np.all(np.isfinite(result[0].data))
        assert np.all(np.isfinite(result[1].data))

    def test_equations_for_metadata(self):
        """Test getting equations with parameter substitution."""
        pde = GrayScottPDE()
        eqs = pde.get_equations_for_metadata(
            {"F": 0.04, "k": 0.06, "Du": 0.16, "Dv": 0.08}
        )

        assert "u" in eqs
        assert "v" in eqs
        # Parameters should be substituted
        assert "0.04" in eqs["u"]
        assert "0.06" in eqs["v"]
