"""Tests for blob diffusion heat equation PDE."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.basic.heat import BlobDiffusionHeatPDE

from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
)


class TestBlobDiffusionHeatPDE:
    """Tests for the blob diffusion heat equation preset."""

    def test_registered(self):
        """Test that blob-diffusion-heat is registered."""
        assert "blob-diffusion-heat" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("blob-diffusion-heat")
        meta = preset.metadata

        assert meta.name == "blob-diffusion-heat"
        assert meta.category == "basic"
        assert meta.num_fields == 1
        assert "T" in meta.field_names

    def test_create_pde(self):
        """Test PDE creation with blob-based diffusion."""
        preset = get_pde_preset("blob-diffusion-heat")
        grid = create_grid_for_dimension(2, resolution=16, periodic=False)
        bc = create_bc_for_dimension(2, periodic=False)
        params = {"D_min": 0.1, "D_max": 1.0, "n_blobs": 5, "sigma": 0.1, "seed": 42}

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None
        # Check that g, dg_dx, dg_dy are in consts
        assert "g" in pde.consts
        assert "dg_dx" in pde.consts
        assert "dg_dy" in pde.consts

    def test_diffusion_coefficient_bounds(self):
        """Test that diffusion coefficient is within expected bounds."""
        preset = get_pde_preset("blob-diffusion-heat")
        grid = create_grid_for_dimension(2, resolution=32, periodic=False)
        bc = create_bc_for_dimension(2, periodic=False)
        params = {"D_min": 0.1, "D_max": 2.0, "n_blobs": 5, "sigma": 0.1, "seed": 42}

        pde = preset.create_pde(params, bc, grid)

        g_field = pde.consts["g"]
        # g should be >= D_min everywhere
        assert np.all(g_field.data >= params["D_min"] - 1e-10)

    def test_short_simulation(self):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = BlobDiffusionHeatPDE()
        grid = create_grid_for_dimension(2, resolution=16, periodic=False)
        bc = create_bc_for_dimension(2, periodic=False)

        params = {"D_min": 0.1, "D_max": 1.0, "n_blobs": 3, "sigma": 0.1, "seed": 42}
        pde = preset.create_pde(params, bc, grid)

        # Create uniform initial condition
        state = ScalarField.from_expression(grid, "1.0")
        state.label = "T"

        result = pde.solve(state, t_range=0.005, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()

    def test_dimension_support_2d_only(self):
        """Test that blob-diffusion-heat only supports 2D."""
        preset = BlobDiffusionHeatPDE()
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)
