"""Tests for wave equation PDEs."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.wave import WavePDE, InhomogeneousWavePDE

from tests.conftest import run_short_simulation
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


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

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("wave", "basic", t_end=0.01)

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "wave"

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_dimension_support(self, ndim: int):
        """Test wave equation works in all supported dimensions."""
        np.random.seed(42)
        preset = WavePDE()

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
        assert isinstance(result, FieldCollection)
        check_result_finite(result, "wave", ndim)
        check_dimension_variation(result, ndim, "wave")


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

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("inhomogeneous-wave", "basic", t_end=0.01)

        assert isinstance(result, FieldCollection)
        assert len(result) == 2
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "inhomogeneous-wave"

    def test_dimension_support_2d_only(self):
        """Test that inhomogeneous-wave only supports 2D."""
        preset = InhomogeneousWavePDE()
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)
