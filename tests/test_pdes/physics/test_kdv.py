"""Tests for Korteweg-de Vries (KdV) equation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.kdv import KdVPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


@pytest.fixture
def small_grid():
    """Create a small 1D grid for fast tests (KdV is a 1D equation)."""
    return CartesianGrid([[0, 10]], [64], periodic=True)


class TestKdVPDE:
    """Tests for Korteweg-de Vries equation."""

    def test_registered(self):
        """Test that kdv is registered."""
        assert "kdv" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("kdv")
        meta = preset.metadata

        assert meta.name == "kdv"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names
        assert "Korteweg" in meta.reference or "KdV" in meta.description

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("kdv")
        params = preset.get_default_parameters()

        assert "b" in params
        assert params["b"] == 0.0001

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("kdv")
        params = preset.get_default_parameters()
        bc = {"x-": "periodic", "x+": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        assert pde is not None

    def test_create_initial_state_soliton(self, small_grid):
        """Test creating initial state with single soliton."""
        preset = get_pde_preset("kdv")
        state = preset.create_initial_state(
            small_grid, "soliton", {"k": 0.5, "x0": 5.0}
        )

        assert state is not None
        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()
        # Soliton should have positive values
        assert state.data.max() > 0

    def test_create_initial_state_two_solitons(self, small_grid):
        """Test creating initial state with two solitons."""
        preset = get_pde_preset("kdv")
        state = preset.create_initial_state(
            small_grid, "two-solitons", {"k1": 0.6, "k2": 0.4}
        )

        assert state is not None
        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()

    def test_create_initial_state_n_wave(self, small_grid):
        """Test creating initial state with n-wave."""
        preset = get_pde_preset("kdv")
        state = preset.create_initial_state(
            small_grid, "n-wave", {"amplitude": 1.0, "width": 0.1}
        )

        assert state is not None
        assert isinstance(state, ScalarField)
        assert np.isfinite(state.data).all()

    def test_soliton_amplitude_width_relation(self, small_grid):
        """Test that soliton amplitude follows u = 2k^2."""
        preset = get_pde_preset("kdv")
        k = 0.5
        state = preset.create_initial_state(
            small_grid, "soliton", {"k": k, "x0": 5.0}
        )

        expected_amplitude = 2 * k**2
        # Max value should be close to expected amplitude
        assert abs(state.data.max() - expected_amplitude) < 0.1 * expected_amplitude

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("kdv", "physics", t_end=0.001)

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "kdv"

    def test_equations_for_metadata(self):
        """Test that equations are properly formatted."""
        preset = get_pde_preset("kdv")
        params = {"b": 0.0001}
        equations = preset.get_equations_for_metadata(params)

        assert "u" in equations
        # Should contain derivative terms (now uses mathematical notation)
        assert "dx" in equations["u"]
        # Should contain the nonlinear term (6*u)
        assert "6*u" in equations["u"]

    def test_dimension_support_1d_only(self):
        """Test KdV equation only supports 1D (true 1D equation)."""
        np.random.seed(42)
        preset = KdVPDE()

        # Check only 1D is supported
        assert preset.metadata.supported_dimensions == [1]

        # Should accept 1D
        preset.validate_dimension(1)

        # Should reject 2D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(2)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)

    def test_dimension_1d_simulation(self):
        """Test KdV simulation in 1D."""
        np.random.seed(42)
        preset = KdVPDE()

        # Create 1D grid and BCs
        grid = create_grid_for_dimension(1, resolution=64)
        bc = create_bc_for_dimension(1)

        # Create PDE and initial state
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation (3rd order derivative - can be stiff)
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "kdv", 1)
