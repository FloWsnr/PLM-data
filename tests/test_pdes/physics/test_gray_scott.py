"""Tests for Gray-Scott PDE."""

import numpy as np
import pytest

from pde import PDE, CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.gray_scott import GrayScottPDE

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
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


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
        # New parameters: a, b, D (3 parameters)
        assert len(meta.parameters) == 3

    def test_create_pde(self, small_grid):
        """Test creating the PDE object."""
        pde_preset = GrayScottPDE()
        pde = pde_preset.create_pde(
            parameters={"a": 0.037, "b": 0.06, "D": 2.0},
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
            ic_type="gaussian-blob",
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
        # u should start mostly at 0.0 (with small perturbation)
        assert np.mean(state[0].data) < 0.2
        # v should start mostly at 1.0
        assert np.mean(state[1].data) > 0.8

    def test_registered_in_registry(self):
        """Test that gray-scott PDE is registered."""
        presets = list_presets()
        assert "gray-scott" in presets

        # Can retrieve via registry
        pde = get_pde_preset("gray-scott")
        assert isinstance(pde, GrayScottPDE)

    def test_short_simulation(self):
        """Test running a short simulation with Gray-Scott using default config."""
        result, config = run_short_simulation("gray-scott", "physics")

        # Result should be a FieldCollection
        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        # Values should be finite
        assert np.all(np.isfinite(result[0].data))
        assert np.all(np.isfinite(result[1].data))
        assert config["preset"] == "gray-scott"

    def test_equations_for_metadata(self):
        """Test getting equations with parameter substitution."""
        pde = GrayScottPDE()
        eqs = pde.get_equations_for_metadata(
            {"a": 0.037, "b": 0.06, "D": 2.0}
        )

        assert "u" in eqs
        assert "v" in eqs
        # Parameters should be substituted
        assert "0.037" in eqs["u"]
        assert "0.06" in eqs["u"] or "0.06" in eqs["v"]

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test Gray-Scott works in all supported dimensions."""
        np.random.seed(42)
        preset = GrayScottPDE()

        # Check dimension is supported
        assert ndim in preset.metadata.supported_dimensions
        preset.validate_dimension(ndim)

        # Create grid and BCs
        resolution = 8 if ndim == 3 else 16
        grid = create_grid_for_dimension(ndim, resolution=resolution)
        bc = create_bc_for_dimension(ndim)

        # Create PDE and initial state
        pde = preset.create_pde({"a": 0.037, "b": 0.06, "D": 2.0}, bc, grid)
        state = preset.create_initial_state(grid, "random-uniform", {"low": 0.1, "high": 0.9})

        # Run short simulation
        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        # Verify result
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "gray-scott", ndim)
        check_dimension_variation(result, ndim, "gray-scott")
