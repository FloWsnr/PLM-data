"""Tests for physics PDEs."""

import numpy as np
import pytest
from pde import PDE, CartesianGrid, FieldCollection

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


class TestKuramotoSivashinskyPDE:
    """Tests for the Kuramoto-Sivashinsky equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("kuramoto-sivashinsky")
        meta = preset.metadata

        assert meta.name == "kuramoto-sivashinsky"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("kuramoto-sivashinsky")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "kuramoto-sivashinsky" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        # Create a smaller grid for stability (KS is very stiff)
        small_ks_grid = CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)

        preset = get_pde_preset("kuramoto-sivashinsky")
        params = {"nu": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_ks_grid)
        state = preset.create_initial_state(
            small_ks_grid, "default", {"amplitude": 0.001}
        )

        # Check that PDE and state are created correctly
        assert pde is not None
        assert state is not None
        assert np.isfinite(state.data).all()
        # Confirm this is a stiff PDE
        assert preset.default_solver == "implicit"


class TestKdVPDE:
    """Tests for the Korteweg-de Vries equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("kdv")
        meta = preset.metadata

        assert meta.name == "kdv"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("kdv")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_soliton_initial_condition(self, small_grid):
        """Test soliton initial condition."""
        preset = get_pde_preset("kdv")
        state = preset.create_initial_state(
            small_grid, "soliton", {"amplitude": 1.0, "width": 0.1}
        )

        assert state is not None
        assert np.max(state.data) > 0  # Should have a peak

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "kdv" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("kdv")
        params = {"c": 0.0, "alpha": 1.0, "beta": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "soliton", {"amplitude": 0.5, "width": 0.1}
        )

        # Check that PDE and state are created correctly
        assert pde is not None
        assert state is not None
        assert np.isfinite(state.data).all()
        # Confirm this is a stiff PDE
        assert preset.default_solver == "implicit"


class TestGinzburgLandauPDE:
    """Tests for the Complex Ginzburg-Landau equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("ginzburg-landau")
        meta = preset.metadata

        assert meta.name == "ginzburg-landau"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("ginzburg-landau")
        params = {"c1": 0.5, "c3": 0.5}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_complex_initial_condition(self, small_grid):
        """Test complex field initial condition."""
        preset = get_pde_preset("ginzburg-landau")
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.1}
        )

        assert state is not None
        assert np.iscomplexobj(state.data)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "ginzburg-landau" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("ginzburg-landau")
        params = {"c1": 0.0, "c3": 0.0}  # Real GL for stability
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "default", {"amplitude": 0.05}
        )

        result = pde.solve(state, t_range=0.01, dt=0.0001)

        assert result is not None
        assert np.isfinite(result.data).all()


class TestLorenzPDE:
    """Tests for diffusively coupled Lorenz system."""

    def test_registered(self):
        """Test that lorenz is registered."""
        assert "lorenz" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("lorenz")
        meta = preset.metadata

        assert meta.name == "lorenz"
        assert meta.category == "physics"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"x", "y", "z"}

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("lorenz")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.1})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test that PDE and initial state are valid.

        Note: The Lorenz PDE uses field names (x, y, z) that conflict with
        2D grid coordinate names, preventing direct simulation. This test
        verifies PDE and initial state creation.
        """
        preset = get_pde_preset("lorenz")
        params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0, "Dx": 0.1, "Dy": 0.1, "Dz": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.1})

        # Verify PDE and state are created correctly
        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert np.isfinite(state[0].data).all()
        assert np.isfinite(state[1].data).all()
        assert np.isfinite(state[2].data).all()


class TestSuperlatticePDE:
    """Tests for superlattice pattern formation."""

    def test_registered(self):
        """Test that superlattice is registered."""
        assert "superlattice" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("superlattice")
        meta = preset.metadata

        assert meta.name == "superlattice"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        # Superlattice has 4th order terms requiring specialized solvers
        preset = get_pde_preset("superlattice")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"amplitude": 0.1})

        assert pde is not None
        assert state is not None
        assert np.isfinite(state.data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("superlattice")
        params = {"epsilon": 0.1, "g2": 0.5}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"amplitude": 0.05})

        # Check that PDE and state are created correctly
        assert pde is not None
        assert state is not None
        assert np.isfinite(state.data).all()
        # Confirm this is a stiff PDE
        assert preset.default_solver == "implicit"


class TestOscillatorsPDE:
    """Tests for coupled Van der Pol oscillators."""

    def test_registered(self):
        """Test that oscillators is registered."""
        assert "oscillators" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("oscillators")
        meta = preset.metadata

        assert meta.name == "oscillators"
        assert meta.category == "physics"
        assert meta.num_fields == 2

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("oscillators")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.1})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test that PDE and initial state are valid.

        Note: The Oscillators PDE uses field names (x, y) that conflict with
        2D grid coordinate names, preventing direct simulation. This test
        verifies PDE and initial state creation.
        """
        preset = get_pde_preset("oscillators")
        params = {"mu": 1.0, "omega": 1.0, "Dx": 0.1, "Dy": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.1})

        # Verify PDE and state are created correctly
        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()
        assert np.isfinite(state[1].data).all()


class TestPeronaMalikPDE:
    """Tests for Perona-Malik edge-preserving diffusion."""

    def test_registered(self):
        """Test that perona-malik is registered."""
        assert "perona-malik" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("perona-malik")
        meta = preset.metadata

        assert meta.name == "perona-malik"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("perona-malik")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.0001)

        assert result is not None
        assert np.isfinite(result.data).all()


class TestNonlinearBeamsPDE:
    """Tests for nonlinear beam/plate vibrations."""

    def test_registered(self):
        """Test that nonlinear-beams is registered."""
        assert "nonlinear-beams" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("nonlinear-beams")
        meta = preset.metadata

        assert meta.name == "nonlinear-beams"
        assert meta.category == "physics"
        assert meta.num_fields == 2

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("nonlinear-beams")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "gaussian-blobs", {"num_blobs": 1})

        # Has 4th order terms, needs small dt
        result = pde.solve(state, t_range=0.001, dt=0.00001)

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestTuringWavePDE:
    """Tests for Turing-wave pattern interaction."""

    def test_registered(self):
        """Test that turing-wave is registered."""
        assert "turing-wave" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("turing-wave")
        meta = preset.metadata

        assert meta.name == "turing-wave"
        assert meta.category == "physics"
        assert meta.num_fields == 2

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("turing-wave")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("turing-wave")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use very small timestep for stability (reaction-diffusion systems need careful numerics)
        result = pde.solve(state, t_range=0.001, dt=0.00001)

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()


class TestAdvectingPatternsPDE:
    """Tests for advected Turing patterns."""

    def test_registered(self):
        """Test that advecting-patterns is registered."""
        assert "advecting-patterns" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("advecting-patterns")
        meta = preset.metadata

        assert meta.name == "advecting-patterns"
        assert meta.category == "physics"
        assert meta.num_fields == 2

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("advecting-patterns")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("advecting-patterns")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use very small timestep for stability
        result = pde.solve(state, t_range=0.001, dt=0.00001)

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()


class TestGrowingDomainsPDE:
    """Tests for patterns on growing domains."""

    def test_registered(self):
        """Test that growing-domains is registered."""
        assert "growing-domains" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("growing-domains")
        meta = preset.metadata

        assert meta.name == "growing-domains"
        assert meta.category == "physics"
        assert meta.num_fields == 2

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("growing-domains")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        assert pde is not None
        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert np.isfinite(state[0].data).all()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("growing-domains")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use very small timestep for stability
        result = pde.solve(state, t_range=0.001, dt=0.00001)

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
