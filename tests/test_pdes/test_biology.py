"""Tests for biology PDEs."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition
from pde_sim.pdes import get_pde_preset, list_presets


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestGiererMeinhardtPDE:
    """Tests for the Gierer-Meinhardt activator-inhibitor system."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("gierer-meinhardt")
        meta = preset.metadata

        assert meta.name == "gierer-meinhardt"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("gierer-meinhardt")
        params = preset.get_default_parameters()

        assert "Du" in params
        assert "Dv" in params
        assert "rho" in params
        assert params["Dv"] > params["Du"]  # Inhibitor should diffuse faster

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("gierer-meinhardt")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test initial state creation."""
        preset = get_pde_preset("gierer-meinhardt")
        state = preset.create_initial_state(
            small_grid, "default", {"noise": 0.01}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "v"
        # All values should be positive
        assert np.all(state[0].data > 0)
        assert np.all(state[1].data > 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "gierer-meinhardt" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("gierer-meinhardt")
        params = {"Du": 0.01, "Dv": 1.0, "rho": 1.0, "mu_u": 0.1, "mu_v": 0.1, "rho_u": 0.01}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.01})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()


class TestKellerSegelPDE:
    """Tests for the Keller-Segel chemotaxis model."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("keller-segel")
        meta = preset.metadata

        assert meta.name == "keller-segel"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names  # cells
        assert "c" in meta.field_names  # chemoattractant

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("keller-segel")
        params = preset.get_default_parameters()

        assert "Du" in params
        assert "Dc" in params
        assert "chi" in params  # chemotactic sensitivity
        assert "alpha" in params  # production
        assert "beta" in params  # decay

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("keller-segel")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_default(self, small_grid):
        """Test default initial state creation."""
        preset = get_pde_preset("keller-segel")
        state = preset.create_initial_state(
            small_grid, "keller-segel-default", {}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        assert state[0].label == "u"
        assert state[1].label == "c"

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "keller-segel" in list_presets()

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("keller-segel")
        # Use moderate parameters to avoid blow-up
        params = {"Du": 1.0, "Dc": 1.0, "chi": 0.5, "alpha": 0.5, "beta": 1.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "keller-segel-default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()


class TestSchnakenbergPDE:
    """Tests for Schnakenberg PDE."""

    def test_registered(self):
        """Test that schnakenberg is registered."""
        assert "schnakenberg" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("schnakenberg")
        meta = preset.metadata

        assert meta.name == "schnakenberg"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("schnakenberg")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use very small timestep for stability
        result = pde.solve(state, t_range=0.0001, dt=0.00001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestBrusselatorPDE:
    """Tests for Brusselator PDE."""

    def test_registered(self):
        """Test that brusselator is registered."""
        assert "brusselator" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("brusselator")
        meta = preset.metadata

        assert meta.name == "brusselator"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("brusselator")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use smaller timestep for stability
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestFisherKPPPDE:
    """Tests for Fisher-KPP PDE."""

    def test_registered(self):
        """Test that fisher-kpp is registered."""
        assert "fisher-kpp" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fisher-kpp")
        meta = preset.metadata

        assert meta.name == "fisher-kpp"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("fisher-kpp")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "random-uniform", {"min_val": 0.0, "max_val": 1.0})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestFitzHughNagumoPDE:
    """Tests for FitzHugh-Nagumo PDE."""

    def test_registered(self):
        """Test that fitzhugh-nagumo is registered."""
        assert "fitzhugh-nagumo" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fitzhugh-nagumo")
        meta = preset.metadata

        assert meta.name == "fitzhugh-nagumo"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("fitzhugh-nagumo")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestAllenCahnPDE:
    """Tests for Allen-Cahn PDE."""

    def test_registered(self):
        """Test that allen-cahn is registered."""
        assert "allen-cahn" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("allen-cahn")
        meta = preset.metadata

        assert meta.name == "allen-cahn"
        assert meta.category == "biology"

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("allen-cahn")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "random-uniform", {"min_val": -1.0, "max_val": 1.0})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestCyclicCompetitionPDE:
    """Tests for cyclic competition (rock-paper-scissors) model."""

    def test_registered(self):
        """Test that cyclic-competition is registered."""
        assert "cyclic-competition" in list_presets()

    def test_metadata(self, small_grid):
        """Test metadata."""
        preset = get_pde_preset("cyclic-competition")
        meta = preset.metadata

        assert meta.name == "cyclic-competition"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert set(meta.field_names) == {"u", "v", "w"}

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("cyclic-competition")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestVegetationPDE:
    """Tests for Klausmeier vegetation model."""

    def test_registered(self):
        """Test that vegetation is registered."""
        assert "vegetation" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("vegetation")
        meta = preset.metadata

        assert meta.name == "vegetation"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "w" in meta.field_names  # water
        assert "n" in meta.field_names  # vegetation

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("vegetation")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestCrossDiffusionPDE:
    """Tests for cross-diffusion model."""

    def test_registered(self):
        """Test that cross-diffusion is registered."""
        assert "cross-diffusion" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cross-diffusion")
        meta = preset.metadata

        assert meta.name == "cross-diffusion"
        assert meta.category == "biology"
        assert meta.num_fields == 2

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("cross-diffusion")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestImmunotherapyPDE:
    """Tests for tumor-immune interaction model."""

    def test_registered(self):
        """Test that immunotherapy is registered."""
        assert "immunotherapy" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("immunotherapy")
        meta = preset.metadata

        assert meta.name == "immunotherapy"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "T" in meta.field_names  # tumor
        assert "I" in meta.field_names  # immune

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("immunotherapy")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestHarshEnvironmentPDE:
    """Tests for Allee effect model."""

    def test_registered(self):
        """Test that harsh-environment is registered."""
        assert "harsh-environment" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("harsh-environment")
        meta = preset.metadata

        assert meta.name == "harsh-environment"
        assert meta.category == "biology"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("harsh-environment")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestBacteriaFlowPDE:
    """Tests for bacteria chemotaxis with flow model."""

    def test_registered(self):
        """Test that bacteria-flow is registered."""
        assert "bacteria-flow" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("bacteria-flow")
        meta = preset.metadata

        assert meta.name == "bacteria-flow"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "b" in meta.field_names  # bacteria
        assert "c" in meta.field_names  # chemical

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("bacteria-flow")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()


class TestHeterogeneousPDE:
    """Tests for spatially heterogeneous model."""

    def test_registered(self):
        """Test that heterogeneous is registered."""
        assert "heterogeneous" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("heterogeneous")
        meta = preset.metadata

        assert meta.name == "heterogeneous"
        assert meta.category == "biology"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("heterogeneous")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        # Use standard IC since heterogeneous doesn't have custom create_initial_state
        state = create_initial_condition(small_grid, "random-uniform", {"min_val": 0.1, "max_val": 0.9})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestTopographyPDE:
    """Tests for population on terrain model."""

    def test_registered(self):
        """Test that topography is registered."""
        assert "topography" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("topography")
        meta = preset.metadata

        assert meta.name == "topography"
        assert meta.category == "biology"
        assert meta.num_fields == 1

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("topography")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "random-uniform", {"min_val": 0.1, "max_val": 0.9})

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()


class TestTuringConditionsPDE:
    """Tests for Turing instability demo model."""

    def test_registered(self):
        """Test that turing-conditions is registered."""
        assert "turing-conditions" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("turing-conditions")
        meta = preset.metadata

        assert meta.name == "turing-conditions"
        assert meta.category == "biology"
        assert meta.num_fields == 2

    def test_create_and_initial_state(self, small_grid):
        """Test that PDE and initial state can be created."""
        preset = get_pde_preset("turing-conditions")
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
        preset = get_pde_preset("turing-conditions")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(small_grid, "default", {"noise": 0.05})

        # Use smaller timestep for stability
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler")

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
