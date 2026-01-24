"""Tests for Vorticity PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestVorticityPDE:
    """Tests for the Vorticity PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("vorticity")
        meta = preset.metadata

        assert meta.name == "vorticity"
        assert meta.category == "fluids"
        assert meta.num_fields == 3
        assert "omega" in meta.field_names
        assert "psi" in meta.field_names
        assert "S" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("vorticity")
        params = {"nu": 0.01, "epsilon": 0.1, "D": 0.0}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state_vortex_pair(self, small_grid):
        """Test vortex pair initial condition."""
        preset = get_pde_preset("vorticity")
        state = preset.create_initial_state(
            small_grid, "vortex-pair", {"strength": 5.0, "radius": 0.1}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        omega = state[0]
        # Should have positive and negative regions (counter-rotating)
        assert np.any(omega.data > 0)
        assert np.any(omega.data < 0)

    def test_registered_in_registry(self):
        """Test that PDE is registered."""
        assert "vorticity" in list_presets()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("vorticity", "fluids")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
        assert config["preset"] == "vorticity"

    def test_dimension_support_2d_only(self):
        """Test that vorticity only supports 2D."""
        preset = get_pde_preset("vorticity")

        # Verify only 2D is supported
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
