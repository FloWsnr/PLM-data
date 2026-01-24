"""Tests for potential flow dipoles PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)


class TestPotentialFlowDipolesPDE:
    """Tests for potential flow dipoles."""

    def test_registered(self):
        """Test that potential-flow-dipoles is registered."""
        assert "potential-flow-dipoles" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("potential-flow-dipoles")
        meta = preset.metadata

        assert meta.name == "potential-flow-dipoles"
        assert meta.category == "fluids"
        assert meta.num_fields == 1  # Only phi field
        assert "phi" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("potential-flow-dipoles")
        params = {"strength": 1.0, "separation": 0.5, "sigma": 0.2, "omega": 1.0, "orbit_radius": 1.0}
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test dipole initial condition."""
        preset = get_pde_preset("potential-flow-dipoles")
        state = preset.create_initial_state(
            small_grid, "default", {}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 1  # Only phi field
        phi = state[0]
        # Potential starts at 0 (evolves as sources move)
        assert np.isfinite(phi.data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("potential-flow-dipoles", "fluids")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
        assert config["preset"] == "potential-flow-dipoles"

    def test_dimension_support_2d_only(self):
        """Test that potential-flow-dipoles only supports 2D."""
        preset = get_pde_preset("potential-flow-dipoles")

        # Verify only 2D is supported
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
