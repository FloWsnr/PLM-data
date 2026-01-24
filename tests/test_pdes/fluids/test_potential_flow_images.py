"""Tests for potential flow with method of images PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=False)


class TestPotentialFlowImagesPDE:
    """Tests for potential flow with method of images."""

    def test_registered(self):
        """Test that potential-flow-images is registered."""
        assert "potential-flow-images" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("potential-flow-images")
        meta = preset.metadata

        assert meta.name == "potential-flow-images"
        assert meta.category == "fluids"
        assert meta.num_fields == 2
        assert "phi" in meta.field_names
        assert "s" in meta.field_names

    def test_create_pde(self, small_grid):
        """Test PDE creation."""
        preset = get_pde_preset("potential-flow-images")
        params = {"strength": 1.0, "wall_x": -2.0, "sigma": 0.5, "omega": 0.5, "amplitude": 0.5}
        bc = {"x": "neumann", "y": "neumann"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None

    def test_create_initial_state(self, small_grid):
        """Test source with image initial condition."""
        preset = get_pde_preset("potential-flow-images")
        state = preset.create_initial_state(
            small_grid, "source-with-image", {"strength": 10.0}
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 2
        phi = state[0]
        # Potential should have variation (source + image)
        assert np.std(phi.data) > 0
        assert np.isfinite(phi.data).all()

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("potential-flow-images", "fluids")

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        for field in result:
            assert np.isfinite(field.data).all()
        assert config["preset"] == "potential-flow-images"

    def test_dimension_support_2d_only(self):
        """Test that potential-flow-images only supports 2D."""
        preset = get_pde_preset("potential-flow-images")

        # Verify only 2D is supported
        assert preset.metadata.supported_dimensions == [2]

        # Should accept 2D
        preset.validate_dimension(2)

        # Should reject 1D and 3D
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)
