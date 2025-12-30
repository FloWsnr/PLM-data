"""Tests for fluid dynamics PDEs."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestVorticityPDE:
    """Tests for the Vorticity PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("vorticity")
        meta = preset.metadata

        assert meta.name == "vorticity"
        assert meta.category == "fluids"
        assert meta.num_fields == 1
        assert meta.field_names == ["w"]
        assert len(meta.parameters) >= 1

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("vorticity")
        params = preset.get_default_parameters()

        assert "nu" in params
        assert params["nu"] > 0

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("vorticity")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state_vortex_pair(self, small_grid):
        """Test vortex pair initial condition."""
        preset = get_pde_preset("vorticity")
        state = preset.create_initial_state(
            small_grid, "vortex-pair", {"strength": 5.0, "radius": 0.1}
        )

        assert isinstance(state, ScalarField)
        assert state.data.shape == small_grid.shape
        # Should have positive and negative regions (counter-rotating)
        assert np.any(state.data > 0)
        assert np.any(state.data < 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "vorticity" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("vorticity")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "single-vortex", {})

        # Run for a few steps
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()


class TestShallowWaterPDE:
    """Tests for the Shallow Water PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("shallow-water")
        meta = preset.metadata

        assert meta.name == "shallow-water"
        assert meta.category == "fluids"
        assert meta.num_fields == 1
        assert meta.field_names == ["h"]

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("shallow-water")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state_drop(self, small_grid):
        """Test water drop initial condition."""
        preset = get_pde_preset("shallow-water")
        state = preset.create_initial_state(
            small_grid, "drop", {"amplitude": 0.5, "background": 1.0}
        )

        assert isinstance(state, ScalarField)
        # Should have values around background with peak at center
        assert np.max(state.data) > 1.0
        assert np.min(state.data) >= 1.0 - 0.01  # Near background

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "shallow-water" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("shallow-water")
        params = {"c": 0.5}  # Slower wave speed for stability
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "drop", {"amplitude": 0.1})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()


class TestNavierStokesPDE:
    """Tests for 2D Navier-Stokes vorticity equation."""

    def test_registered(self):
        """Test that navier-stokes is registered."""
        assert "navier-stokes" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("navier-stokes")
        meta = preset.metadata

        assert meta.name == "navier-stokes"
        assert meta.category == "fluids"
        assert meta.num_fields == 1
        assert "w" in meta.field_names

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("navier-stokes")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "vortex-pair", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestThermalConvectionPDE:
    """Tests for Rayleigh-Bénard thermal convection."""

    def test_registered(self):
        """Test that thermal-convection is registered."""
        assert "thermal-convection" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("thermal-convection")
        meta = preset.metadata

        assert meta.name == "thermal-convection"
        assert meta.category == "fluids"
        assert meta.num_fields == 2
        assert "T" in meta.field_names
        assert "w" in meta.field_names

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("thermal-convection")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestMethodOfImagesPDE:
    """Tests for vortex near wall (method of images)."""

    def test_registered(self):
        """Test that method-of-images is registered."""
        assert "method-of-images" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("method-of-images")
        meta = preset.metadata

        assert meta.name == "method-of-images"
        assert meta.category == "fluids"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("method-of-images")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestDipolesPDE:
    """Tests for vortex dipole dynamics."""

    def test_registered(self):
        """Test that dipoles is registered."""
        assert "dipoles" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("dipoles")
        meta = preset.metadata

        assert meta.name == "dipoles"
        assert meta.category == "fluids"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("dipoles")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        # Dipole should have positive and negative regions
        assert np.any(result.data > 0)
        assert np.any(result.data < 0)
