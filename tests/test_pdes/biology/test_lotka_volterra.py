"""Tests for Lotka-Volterra predator-prey PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestLotkaVolterraPDE:
    """Tests for Lotka-Volterra predator-prey model."""

    def test_registered(self):
        """Test that lotka-volterra is registered."""
        assert "lotka-volterra" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("lotka-volterra")
        meta = preset.metadata

        assert meta.name == "lotka-volterra"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert set(meta.field_names) == {"u", "v"}

        # Check all required parameters are present
        param_names = {p.name for p in meta.parameters}
        assert param_names == {"Du", "Dv", "alpha", "beta", "delta", "gamma"}

    def test_equations(self):
        """Test that equations are properly defined."""
        preset = get_pde_preset("lotka-volterra")
        meta = preset.metadata

        assert "u" in meta.equations
        assert "v" in meta.equations
        assert "laplace(u)" in meta.equations["u"]
        assert "laplace(v)" in meta.equations["v"]

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("lotka-volterra", "biology", t_end=0.1)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all(), "Prey field contains non-finite values"
        assert np.isfinite(result[1].data).all(), "Predator field contains non-finite values"
        assert config["preset"] == "lotka-volterra"

    def test_populations_non_negative(self):
        """Test that populations remain non-negative (biological constraint)."""
        result, config = run_short_simulation("lotka-volterra", "biology", t_end=0.5)

        prey = result[0].data
        predator = result[1].data

        # Populations should not go negative
        # Note: Small negative values may occur due to numerical errors,
        # so we allow a small tolerance
        assert prey.min() >= -1e-10, f"Prey went negative: min={prey.min()}"
        assert predator.min() >= -1e-10, f"Predator went negative: min={predator.min()}"

    def test_longer_simulation_stability(self):
        """Test simulation remains stable over longer time."""
        result, config = run_short_simulation(
            "lotka-volterra", "biology", t_end=1.0, resolution=64
        )

        # Check for finite values after longer simulation
        assert np.isfinite(result[0].data).all(), "Prey field has non-finite values"
        assert np.isfinite(result[1].data).all(), "Predator field has non-finite values"

        # Check that populations haven't exploded to unreasonable values
        # With default parameters, equilibrium is around u*=6.67, v*=10
        # Allow for some oscillation but not explosion
        assert result[0].data.max() < 1000, "Prey population exploded"
        assert result[1].data.max() < 1000, "Predator population exploded"

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Lotka-Volterra works in supported dimensions.

        Note: This PDE uses a custom initial condition that assumes 2D+ grids,
        so we only test 2D and 3D here.
        """
        np.random.seed(42)
        preset = get_pde_preset("lotka-volterra")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state using default IC
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "default", {"noise": 0.1, "seed": 42})

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "lotka-volterra", ndim)
        check_dimension_variation(result, ndim, "lotka-volterra")
