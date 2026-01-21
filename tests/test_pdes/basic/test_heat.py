"""Tests for heat equation PDEs."""

import numpy as np
import pytest

from pde import PDE, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.core.config import BoundaryConfig
from pde_sim.pdes.basic.heat import (
    HeatPDE,
    InhomogeneousHeatPDE,
    InhomogeneousDiffusionHeatPDE,
)

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


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

    def test_short_simulation(self):
        """Test running a short simulation with the heat PDE using default config."""
        result, config = run_short_simulation("heat", "basic", t_end=0.1)

        # Result should be a ScalarField
        assert isinstance(result, ScalarField)
        # Values should be finite
        assert np.all(np.isfinite(result.data))
        # Verify we used the config parameters
        assert config["preset"] == "heat"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test heat equation works in all supported dimensions."""
        np.random.seed(42)
        preset = HeatPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None)

        # Verify result
        assert isinstance(result, ScalarField)
        check_result_finite(result, "heat", ndim)


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

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("inhomogeneous-heat", "basic", t_end=0.01)

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "inhomogeneous-heat"

    def test_dimension_support_2d_only(self):
        """Test that inhomogeneous-heat only supports 2D."""
        preset = InhomogeneousHeatPDE()
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)


class TestInhomogeneousDiffusionHeatPDE:
    """Tests for the Inhomogeneous Diffusion Heat equation preset."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        pde = InhomogeneousDiffusionHeatPDE()
        meta = pde.metadata

        assert meta.name == "inhomogeneous-diffusion-heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert len(meta.parameters) == 3  # D, E, n

    def test_registered_in_registry(self):
        """Test that inhomogeneous-diffusion-heat PDE is registered."""
        presets = list_presets()
        assert "inhomogeneous-diffusion-heat" in presets

    def test_get_default_parameters(self):
        """Test default parameters retrieval."""
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        params = preset.get_default_parameters()

        assert "D" in params
        assert "E" in params
        assert "n" in params
        assert params["D"] == 1.0
        assert params["E"] == 0.99
        assert params["n"] == 40

    def test_create_pde_with_varying_diffusion(self, non_periodic_grid):
        """Test creating PDE with spatially varying diffusion."""
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        pde = preset.create_pde(
            parameters={"D": 1.0, "E": 0.5, "n": 10},
            bc={"x": "dirichlet", "y": "dirichlet"},
            grid=non_periodic_grid,
        )

        assert isinstance(pde, PDE)
        # Check that g, dg_dx, dg_dy are in consts
        assert "g" in pde.consts
        assert "dg_dx" in pde.consts
        assert "dg_dy" in pde.consts

    def test_diffusion_coefficient_positive(self, non_periodic_grid):
        """Test that diffusion coefficient g(x,y) is always positive."""
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        params = {"D": 1.0, "E": 0.99, "n": 40}
        bc = {"x": "dirichlet", "y": "dirichlet"}

        pde = preset.create_pde(params, bc, non_periodic_grid)

        g_field = pde.consts["g"]
        # With E < 1, g should always be positive: D*(1-E) <= g <= D*(1+E)
        assert np.all(g_field.data > 0)
        # Check bounds
        assert np.min(g_field.data) >= params["D"] * (1 - params["E"]) - 1e-10
        assert np.max(g_field.data) <= params["D"] * (1 + params["E"]) + 1e-10

    def test_short_simulation(self, non_periodic_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        params = {"D": 0.1, "E": 0.5, "n": 5}
        bc = BoundaryConfig(
            x_minus="dirichlet:0", x_plus="dirichlet:0",
            y_minus="dirichlet:0", y_plus="dirichlet:0"
        )

        pde = preset.create_pde(params, bc, non_periodic_grid)

        # Use uniform initial condition as specified in todo.md
        state = ScalarField.from_expression(non_periodic_grid, "1.0")
        state.label = "T"

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()

    def test_get_equations_for_metadata(self):
        """Test equations are properly formatted."""
        preset = get_pde_preset("inhomogeneous-diffusion-heat")
        params = {"D": 2.0, "E": 0.8, "n": 20}

        equations = preset.get_equations_for_metadata(params)

        assert "T" in equations
        assert "g(x,y)" in equations
        assert "div(g(x,y) * grad(T))" in equations["T"]
        # Check parameter values are included
        assert "2.0" in equations["g(x,y)"]
        assert "0.8" in equations["g(x,y)"]
        assert "20" in equations["g(x,y)"]

    def test_dimension_support_2d_only(self):
        """Test that inhomogeneous-diffusion-heat only supports 2D."""
        preset = InhomogeneousDiffusionHeatPDE()
        assert preset.metadata.supported_dimensions == [2]

        preset.validate_dimension(2)

        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)
