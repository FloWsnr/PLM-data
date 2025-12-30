"""Tests for basic PDEs."""

import numpy as np
import pytest
from pde import PDE, ScalarField, FieldCollection

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

        # Run short simulation with explicit euler solver
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None)

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

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        np.random.seed(42)
        pde_preset = InhomogeneousHeatPDE()
        params = {"D": 0.01, "source": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = pde_preset.create_pde(params, bc, small_grid)
        state = pde_preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestSchrodingerPDE:
    """Tests for the Schrodinger equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("schrodinger")
        meta = preset.metadata

        assert meta.name == "schrodinger"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "psi" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("schrodinger")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_wave_packet(self, small_grid):
        """Test wave packet initial condition."""
        preset = get_pde_preset("schrodinger")
        state = preset.create_initial_state(
            small_grid, "wave-packet", {"kx": 5.0, "sigma": 0.1}
        )

        assert isinstance(state, ScalarField)
        # Should be complex
        assert np.iscomplexobj(state.data)
        # Should be normalized (approximately)
        norm = np.sum(np.abs(state.data) ** 2)
        assert norm > 0

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "schrodinger" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("schrodinger")
        params = {"D": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "wave-packet", {"kx": 2.0, "sigma": 0.15}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()


class TestPlatePDE:
    """Tests for the biharmonic plate equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("plate")
        meta = preset.metadata

        assert meta.name == "plate"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "u" in meta.field_names
        # Should mention biharmonic/fourth-order
        assert "biharmonic" in meta.description.lower() or "fourth" in meta.description.lower()

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("plate")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "plate" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("plate")
        params = {"D": 0.001}  # Small coefficient for stability
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert result is not None
        assert np.isfinite(result.data).all()


class TestInhomogeneousWavePDE:
    """Tests for the inhomogeneous wave equation with damping."""

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

        assert "c" in params  # wave speed
        assert "gamma" in params  # damping
        assert "source" in params  # forcing

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("inhomogeneous-wave")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test initial state creation."""
        preset = get_pde_preset("inhomogeneous-wave")
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"
        # Velocity should start at zero
        assert np.allclose(state[1].data, 0)

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("inhomogeneous-wave")
        params = {"c": 1.0, "gamma": 0.1, "source": 0.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1}
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()


class TestBackends:
    """Tests for different compute backends."""

    def test_numba_backend(self, small_grid):
        """Test running simulation with numba backend."""
        preset = get_pde_preset("heat")
        params = {"D": 0.01}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        # Test with numba backend
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", backend="numba")

        assert np.all(np.isfinite(result.data))

    def test_numpy_backend(self, small_grid):
        """Test running simulation with numpy backend."""
        preset = get_pde_preset("heat")
        params = {"D": 0.01}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        # Test with numpy backend
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", backend="numpy")

        assert np.all(np.isfinite(result.data))
