"""Tests for Swift-Hohenberg with Advection PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets

from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestSwiftHohenbergAdvectionPDE:
    """Tests for Swift-Hohenberg with Advection PDE."""

    def test_registered(self):
        """Test that swift-hohenberg-advection is registered."""
        assert "swift-hohenberg-advection" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("swift-hohenberg-advection")
        meta = preset.metadata

        assert meta.name == "swift-hohenberg-advection"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("swift-hohenberg-advection", "physics", t_end=0.01)

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "swift-hohenberg-advection"
