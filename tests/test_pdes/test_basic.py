"""Tests for basic PDEs."""

import numpy as np
import pytest
from pde import PDE, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.heat import HeatPDE, InhomogeneousHeatPDE


class TestHeatPDE:
    """Tests for the Heat equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        pde = HeatPDE()
        meta = pde.metadata

        assert meta.name == "heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "T" in meta.field_names
        assert len(meta.parameters) == 1
        assert meta.parameters[0].name == "D"

    def test_get_default_parameters(self):
        """Test getting default parameters."""
        pde = HeatPDE()
        defaults = pde.get_default_parameters()

        assert "D" in defaults
        assert defaults["D"] == 1.0

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid params."""
        pde = HeatPDE()
        # Should not raise
        pde.validate_parameters({"D": 0.5})

    def test_validate_parameters_invalid(self):
        """Test parameter validation with invalid params."""
        pde = HeatPDE()
        with pytest.raises(ValueError, match="D must be >="):
            pde.validate_parameters({"D": 0.0001})

    def test_create_pde(self, small_grid):
        """Test creating the PDE object."""
        pde_preset = HeatPDE()
        pde = pde_preset.create_pde(
            parameters={"D": 0.5},
            bc={"x": "periodic", "y": "periodic"},
            grid=small_grid,
        )

        assert isinstance(pde, PDE)

    def test_create_initial_state(self, small_grid):
        """Test creating initial state."""
        np.random.seed(42)
        pde = HeatPDE()
        state = pde.create_initial_state(
            grid=small_grid,
            ic_type="random-uniform",
            ic_params={"low": 0.0, "high": 1.0},
        )

        assert isinstance(state, ScalarField)
        assert state.data.shape == (32, 32)

    def test_registered_in_registry(self):
        """Test that heat PDE is registered."""
        presets = list_presets()
        assert "heat" in presets

        # Can retrieve via registry
        pde = get_pde_preset("heat")
        assert isinstance(pde, HeatPDE)

    def test_short_simulation(self, small_grid):
        """Test running a short simulation with the heat PDE."""
        np.random.seed(42)
        pde_preset = HeatPDE()

        # Use small diffusion coefficient for stability
        params = {"D": 0.01}
        pde = pde_preset.create_pde(
            parameters=params,
            bc={"x": "periodic", "y": "periodic"},
            grid=small_grid,
        )

        state = pde_preset.create_initial_state(
            grid=small_grid,
            ic_type="gaussian-blobs",
            ic_params={"num_blobs": 1, "amplitude": 1.0},
        )

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.0001, tracker=None)

        # Result should be a ScalarField
        assert isinstance(result, ScalarField)
        # Values should be finite
        assert np.all(np.isfinite(result.data))


class TestInhomogeneousHeatPDE:
    """Tests for the Inhomogeneous Heat equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        pde = InhomogeneousHeatPDE()
        meta = pde.metadata

        assert meta.name == "inhomogeneous-heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert len(meta.parameters) == 2

    def test_registered_in_registry(self):
        """Test that inhomogeneous-heat PDE is registered."""
        presets = list_presets()
        assert "inhomogeneous-heat" in presets

    def test_create_with_source(self, small_grid):
        """Test creating PDE with source term."""
        pde_preset = InhomogeneousHeatPDE()
        pde = pde_preset.create_pde(
            parameters={"D": 0.1, "source": 1.0},
            bc={"x": "periodic", "y": "periodic"},
            grid=small_grid,
        )

        assert isinstance(pde, PDE)
