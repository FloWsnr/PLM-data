"""Tests for basic PDEs."""

import numpy as np
import pytest
from pde import PDE, ScalarField, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.heat import HeatPDE, InhomogeneousHeatPDE
from pde_sim.pdes.basic.wave import WavePDE, AdvectionPDE, InhomogeneousWavePDE
from pde_sim.pdes.basic.schrodinger import SchrodingerPDE, PlatePDE


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
        assert meta.parameters[0].name == "D_T"

    def test_get_default_parameters(self):
        """Test getting default parameters."""
        pde = HeatPDE()
        defaults = pde.get_default_parameters()

        assert "D_T" in defaults
        assert defaults["D_T"] == 1.0  # Updated default from reference

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid params."""
        pde = HeatPDE()
        # Should not raise
        pde.validate_parameters({"D_T": 5.0})

    def test_validate_parameters_invalid(self):
        """Test parameter validation with invalid params."""
        pde = HeatPDE()
        with pytest.raises(ValueError, match="D_T must be >="):
            pde.validate_parameters({"D_T": 0.001})

    def test_create_pde(self, small_grid):
        """Test creating the PDE object."""
        pde_preset = HeatPDE()
        pde = pde_preset.create_pde(
            parameters={"D_T": 1.0},
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
        params = {"D_T": 0.01}
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
        assert len(meta.parameters) == 3  # D, n, m

    def test_registered_in_registry(self):
        """Test that inhomogeneous-heat PDE is registered."""
        presets = list_presets()
        assert "inhomogeneous-heat" in presets

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("inhomogeneous-heat")
        params = preset.get_default_parameters()

        assert "D" in params  # diffusion coefficient
        assert "n" in params  # spatial mode x
        assert "m" in params  # spatial mode y
        assert params["D"] == 1.0
        assert params["n"] == 4
        assert params["m"] == 4

    def test_create_with_source(self, non_periodic_grid):
        """Test creating PDE with spatial source term."""
        pde_preset = InhomogeneousHeatPDE()
        pde = pde_preset.create_pde(
            parameters={"D": 0.1, "n": 2, "m": 2},
            bc={"x": "neumann", "y": "neumann"},
            grid=non_periodic_grid,
        )

        assert isinstance(pde, PDE)

    def test_short_simulation(self, non_periodic_grid):
        """Test running a short simulation."""
        np.random.seed(42)
        pde_preset = InhomogeneousHeatPDE()
        params = {"D": 0.01, "n": 2, "m": 2}
        bc = {"x": "neumann", "y": "neumann"}

        pde = pde_preset.create_pde(params, bc, non_periodic_grid)
        state = pde_preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


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


class TestAdvectionPDE:
    """Tests for the Advection-diffusion equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("advection")
        meta = preset.metadata

        assert meta.name == "advection"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("advection")
        params = preset.get_default_parameters()

        assert "D" in params
        assert "V" in params
        assert "theta" in params
        assert "mode" in params
        assert params["D"] == 1.0
        assert params["V"] == 0.10
        assert params["mode"] == 0  # rotational

    def test_create_pde_rotational(self, small_grid):
        """Test PDE creation with rotational mode."""
        preset = get_pde_preset("advection")
        params = {"D": 1.0, "V": 0.1, "mode": 0}
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_pde_directed(self, small_grid):
        """Test PDE creation with directed mode."""
        preset = get_pde_preset("advection")
        params = {"D": 1.0, "V": 6.0, "theta": -2.0, "mode": 1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "advection" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("advection")
        params = {"D": 0.1, "V": 0.1, "mode": 0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

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


class TestBoundaryConditions:
    """Tests for boundary condition conversion."""

    def test_convert_bc_periodic(self):
        """Test periodic BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "periodic", "y": "periodic"})
        assert bc == ["periodic", "periodic"]

    def test_convert_bc_no_flux(self):
        """Test no-flux (Neumann) BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "no-flux", "y": "no-flux"})
        assert bc == [{"derivative": 0}, {"derivative": 0}]

    def test_convert_bc_dirichlet(self):
        """Test Dirichlet BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "dirichlet", "y": "dirichlet"})
        assert bc == [{"value": 0}, {"value": 0}]

    def test_convert_bc_mixed(self):
        """Test mixed BC conversion."""
        preset = get_pde_preset("heat")
        bc = preset._convert_bc({"x": "periodic", "y": "no-flux"})
        assert bc == ["periodic", {"derivative": 0}]

    def test_simulation_with_no_flux_bc(self, non_periodic_grid):
        """Test running simulation with no-flux boundary conditions."""
        preset = get_pde_preset("heat")
        params = {"D_T": 0.01}
        bc = {"x": "no-flux", "y": "no-flux"}

        pde = preset.create_pde(params, bc, non_periodic_grid)
        state = preset.create_initial_state(
            non_periodic_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert np.all(np.isfinite(result.data))


class TestBackends:
    """Tests for different compute backends."""

    def test_numba_backend(self, small_grid):
        """Test running simulation with numba backend."""
        preset = get_pde_preset("heat")
        params = {"D_T": 0.01}
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
        params = {"D_T": 0.01}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        # Test with numpy backend
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", backend="numpy")

        assert np.all(np.isfinite(result.data))
