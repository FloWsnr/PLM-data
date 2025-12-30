"""Tests for biology PDEs."""

import numpy as np
import pytest
from pde import CartesianGrid, FieldCollection, ScalarField

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

        result = pde.solve(state, t_range=0.1, dt=0.001)

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

        result = pde.solve(state, t_range=0.1, dt=0.001)

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()


class TestSchnakenbergPDE:
    """Basic tests for existing Schnakenberg PDE."""

    def test_registered(self):
        """Test that schnakenberg is registered."""
        assert "schnakenberg" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("schnakenberg")
        meta = preset.metadata

        assert meta.name == "schnakenberg"
        assert meta.category == "biology"


class TestBrusselatorPDE:
    """Basic tests for existing Brusselator PDE."""

    def test_registered(self):
        """Test that brusselator is registered."""
        assert "brusselator" in list_presets()


class TestFisherKPPPDE:
    """Basic tests for existing Fisher-KPP PDE."""

    def test_registered(self):
        """Test that fisher-kpp is registered."""
        assert "fisher-kpp" in list_presets()


class TestFitzHughNagumoPDE:
    """Basic tests for existing FitzHugh-Nagumo PDE."""

    def test_registered(self):
        """Test that fitzhugh-nagumo is registered."""
        assert "fitzhugh-nagumo" in list_presets()


class TestAllenCahnPDE:
    """Basic tests for existing Allen-Cahn PDE."""

    def test_registered(self):
        """Test that allen-cahn is registered."""
        assert "allen-cahn" in list_presets()
