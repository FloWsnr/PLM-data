"""Tests for FitzHugh-Nagumo 3-species PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.biology.fitzhugh_nagumo_3 import FitzHughNagumo3PDE

from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
)


class TestFitzHughNagumo3PDE:
    """Tests for the FitzHugh-Nagumo 3-species equation preset."""

    def test_registered(self):
        """Test that fitzhugh-nagumo-3 is registered."""
        assert "fitzhugh-nagumo-3" in list_presets()

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("fitzhugh-nagumo-3")
        meta = preset.metadata

        assert meta.name == "fitzhugh-nagumo-3"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names

    def test_get_default_parameters(self):
        """Test default parameters."""
        preset = get_pde_preset("fitzhugh-nagumo-3")
        params = preset.get_default_parameters()

        assert "Dv" in params
        assert "Dw" in params
        assert "a_v" in params
        assert "e_v" in params
        assert "e_w" in params
        assert "a_w" in params
        assert "a_z" in params
        assert params["Dv"] == 40.0
        assert params["Dw"] == 200.0

    def test_create_pde(self):
        """Test PDE creation."""
        preset = get_pde_preset("fitzhugh-nagumo-3")
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)
        params = preset.get_default_parameters()

        pde = preset.create_pde(params, bc, grid)
        assert pde is not None

    def test_create_initial_state(self):
        """Test initial state creation."""
        preset = get_pde_preset("fitzhugh-nagumo-3")
        grid = create_grid_for_dimension(2, resolution=16)

        state = preset.create_initial_state(grid, "default", {"seed": 42})

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert state[0].label == "u"
        assert state[1].label == "v"
        assert state[2].label == "w"

    def test_short_simulation(self):
        """Test running a short simulation."""
        np.random.seed(42)
        preset = FitzHughNagumo3PDE()
        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "default", {"seed": 42})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None)

        assert isinstance(result, FieldCollection)
        assert len(result) == 3
        for field in result:
            assert np.isfinite(field.data).all()

    def test_dimension_support_2d_only(self):
        """Test that fitzhugh-nagumo-3 only supports 2D (IC uses 2D meshgrid)."""
        preset = FitzHughNagumo3PDE()
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError):
            preset.validate_dimension(1)
        with pytest.raises(ValueError):
            preset.validate_dimension(3)

    def test_dimension_2d_simulation(self):
        """Test fitzhugh-nagumo-3 simulation in 2D."""
        np.random.seed(42)
        preset = FitzHughNagumo3PDE()

        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        pde = preset.create_pde(preset.get_default_parameters(), bc, grid)
        state = preset.create_initial_state(grid, "default", {"seed": 42})

        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", tracker=None)

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "fitzhugh-nagumo-3", 2)
