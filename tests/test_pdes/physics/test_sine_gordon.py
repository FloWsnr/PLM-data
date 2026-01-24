"""Tests for Sine-Gordon equation PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.sine_gordon import SineGordonPDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestSineGordonPDE:
    """Tests for Sine-Gordon equation PDE."""

    def test_registered(self):
        """Test that sine-gordon is registered."""
        assert "sine-gordon" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("sine-gordon")
        meta = preset.metadata

        assert meta.name == "sine-gordon"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "phi" in meta.field_names
        assert "psi" in meta.field_names

        # Check parameters
        param_names = [p.name for p in meta.parameters]
        assert "c" in param_names
        assert "gamma" in param_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("sine-gordon", "physics", t_end=0.1)

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all(), "phi field contains NaN or Inf"
        assert np.isfinite(result[1].data).all(), "psi field contains NaN or Inf"
        assert config["preset"] == "sine-gordon"

    def test_kink_initial_condition(self, small_grid):
        """Test kink initial condition creates expected profile."""
        preset = get_pde_preset("sine-gordon")
        state = preset.create_initial_state(
            small_grid,
            ic_type="kink",
            ic_params={"width": 1.0},
        )

        phi = state[0].data
        psi = state[1].data

        # Kink should transition from ~0 to ~2*pi
        assert phi.min() >= -0.5  # Close to 0 on left
        assert phi.max() <= 2 * np.pi + 0.5  # Close to 2*pi on right

        # Velocity should be zero for stationary kink
        assert np.allclose(psi, 0, atol=1e-10)

    def test_antikink_initial_condition(self, small_grid):
        """Test antikink initial condition."""
        preset = get_pde_preset("sine-gordon")
        state = preset.create_initial_state(
            small_grid,
            ic_type="antikink",
            ic_params={"width": 1.0},
        )

        phi = state[0].data

        # Antikink should be opposite orientation from kink
        # Values should still be bounded
        assert np.isfinite(phi).all()

    def test_kink_antikink_initial_condition(self, small_grid):
        """Test kink-antikink pair initial condition."""
        preset = get_pde_preset("sine-gordon")
        state = preset.create_initial_state(
            small_grid,
            ic_type="kink-antikink",
            ic_params={"width": 1.0, "v_kink": 0.3, "v_antikink": -0.3},
        )

        phi = state[0].data
        psi = state[1].data

        # Both fields should be finite
        assert np.isfinite(phi).all()
        assert np.isfinite(psi).all()

        # Velocity field should be non-zero (moving solitons)
        assert not np.allclose(psi, 0)

    def test_breather_initial_condition(self, small_grid):
        """Test breather initial condition."""
        preset = get_pde_preset("sine-gordon")
        state = preset.create_initial_state(
            small_grid,
            ic_type="breather",
            ic_params={"width": 2.0, "omega": 0.5},
        )

        phi = state[0].data
        psi = state[1].data

        # Both fields should be finite
        assert np.isfinite(phi).all()
        assert np.isfinite(psi).all()

    def test_ring_initial_condition(self, small_grid):
        """Test ring soliton initial condition."""
        preset = get_pde_preset("sine-gordon")
        state = preset.create_initial_state(
            small_grid,
            ic_type="ring",
            ic_params={"radius": 3.0, "width": 0.5},
        )

        phi = state[0].data

        # Ring should be radially symmetric
        # Check that center and edge have different values
        center = phi[8, 8]
        edge = phi[0, 0]
        assert not np.isclose(center, edge)

    def test_random_initial_condition(self, small_grid):
        """Test random initial condition."""
        preset = get_pde_preset("sine-gordon")
        state = preset.create_initial_state(
            small_grid,
            ic_type="random",
            ic_params={"amplitude": 0.5, "seed": 42},
        )

        phi = state[0].data
        psi = state[1].data

        # phi should have random values
        assert phi.std() > 0

        # psi should be zero for random IC
        assert np.allclose(psi, 0)

    def test_pde_creation(self, small_grid):
        """Test PDE object creation."""
        from pde_sim.core.config import BoundaryConfig

        preset = get_pde_preset("sine-gordon")
        bc = BoundaryConfig(
            x_minus="periodic",
            x_plus="periodic",
            y_minus="periodic",
            y_plus="periodic",
        )

        pde = preset.create_pde(
            parameters={"c": 1.0, "gamma": 0.01},
            bc=bc,
            grid=small_grid,
        )

        # PDE should be created successfully
        assert pde is not None

    def test_equations_for_metadata(self):
        """Test equation string generation."""
        preset = get_pde_preset("sine-gordon")
        eqs = preset.get_equations_for_metadata({"c": 2.0, "gamma": 0.05})

        assert "phi" in eqs
        assert "psi" in eqs
        assert "sin(phi)" in eqs["psi"]
        assert "4.0" in eqs["psi"]  # c^2 = 4.0

    def test_simulation_stability(self):
        """Test that simulation remains stable over longer time."""
        result, config = run_short_simulation(
            "sine-gordon", "physics", t_end=1.0, resolution=64
        )

        # Check that values remain bounded
        phi = result[0].data
        psi = result[1].data

        # phi should remain bounded (not blow up)
        assert np.abs(phi).max() < 100, "phi values grew too large"
        assert np.abs(psi).max() < 100, "psi values grew too large"

    def test_dimension_support_2d_only(self):
        """Test Sine-Gordon equation only supports 2D."""
        np.random.seed(42)
        preset = SineGordonPDE()

        # Check only 2D is supported
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)

    def test_dimension_2d_simulation(self):
        """Test Sine-Gordon simulation in 2D."""
        np.random.seed(42)
        preset = SineGordonPDE()

        # Create 2D grid and BCs
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        # Create PDE and initial state
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random", {"amplitude": 0.5})

        # Run short simulation
        result = pde.solve(state, t_range=0.1, dt=0.01, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "sine-gordon", 2)
        check_dimension_variation(result, 2, "sine-gordon")
