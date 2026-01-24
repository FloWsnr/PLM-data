"""Tests for SIR epidemic model."""

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


class TestSIRPDE:
    """Tests for SIR epidemic model."""

    def test_registered(self):
        """Test that SIR is registered."""
        assert "sir" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("sir")
        meta = preset.metadata

        assert meta.name == "sir"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "S" in meta.field_names  # Susceptible
        assert "I" in meta.field_names  # Infected
        assert "R" in meta.field_names  # Recovered

        # Check parameters
        param_names = [p.name for p in meta.parameters]
        assert "beta" in param_names
        assert "gamma" in param_names
        assert "D_S" in param_names
        assert "D_I" in param_names
        assert "D_R" in param_names

    def test_default_parameters(self):
        """Test default parameter values."""
        preset = get_pde_preset("sir")
        defaults = preset.get_default_parameters()

        assert defaults["beta"] == 0.5
        assert defaults["gamma"] == 0.1
        assert defaults["D_S"] == 0.1
        assert defaults["D_I"] == 0.1
        assert defaults["D_R"] == 0.1

        # Check R0 > 1 for epidemic behavior
        R0 = defaults["beta"] / defaults["gamma"]
        assert R0 > 1, f"R0 = {R0} should be > 1 for epidemic"

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("sir", "biology", t_end=0.1)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all(), "S contains NaN/Inf"
        assert np.isfinite(result[1].data).all(), "I contains NaN/Inf"
        assert np.isfinite(result[2].data).all(), "R contains NaN/Inf"
        assert config["preset"] == "sir"

    def test_population_conservation(self):
        """Test that S + I + R remains approximately constant.

        With equal diffusion and no-flux boundaries, total population
        should be conserved.
        """
        result, config = run_short_simulation(
            "sir", "biology", t_end=0.1, resolution=32
        )

        # Get individual field data
        S = result[0].data
        I = result[1].data
        R = result[2].data

        # Check that S + I + R sums to approximately 1 everywhere
        total = S + I + R
        assert np.allclose(
            total, 1.0, atol=1e-3
        ), f"Population not conserved: min={total.min()}, max={total.max()}"

    def test_non_negative_populations(self):
        """Test that populations remain non-negative."""
        result, config = run_short_simulation(
            "sir", "biology", t_end=0.5, resolution=32
        )

        S = result[0].data
        I = result[1].data
        R = result[2].data

        assert (S >= -1e-10).all(), f"S has negative values: min={S.min()}"
        assert (I >= -1e-10).all(), f"I has negative values: min={I.min()}"
        assert (R >= -1e-10).all(), f"R has negative values: min={R.min()}"

    def test_recovered_increases(self):
        """Test that recovered population generally increases over time."""
        # Run a slightly longer simulation to see epidemic dynamics
        result, config = run_short_simulation(
            "sir", "biology", t_end=1.0, resolution=32
        )

        R = result[2].data

        # Average R should be greater than initial (which was 0)
        assert R.mean() > 0.0, "Recovered population should increase"

    def test_equations_format(self):
        """Test that equations are properly formatted."""
        preset = get_pde_preset("sir")
        meta = preset.metadata

        assert "S" in meta.equations
        assert "I" in meta.equations
        assert "R" in meta.equations

        # Check equation structure contains expected terms
        assert "laplace(S)" in meta.equations["S"]
        assert "beta" in meta.equations["S"]
        assert "laplace(I)" in meta.equations["I"]
        assert "gamma" in meta.equations["I"]
        assert "laplace(R)" in meta.equations["R"]

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_dimension_support(self, ndim: int):
        """Test SIR works in supported dimensions.

        Note: SIR uses a custom initial condition that assumes 2D+ grids,
        so we only test 2D and 3D here.
        """
        np.random.seed(42)
        preset = get_pde_preset("sir")

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state (use random-uniform for variation testing)
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "sir", ndim)
        check_dimension_variation(result, ndim, "sir")
