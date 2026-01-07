"""Tests for Duffing Oscillator PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets

from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestDuffingPDE:
    """Tests for Duffing Oscillator PDE."""

    def test_registered(self):
        """Test that duffing is registered."""
        assert "duffing" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("duffing")
        meta = preset.metadata

        assert meta.name == "duffing"
        assert meta.category == "physics"
        assert meta.num_fields == 2
        assert "X" in meta.field_names
        assert "Y" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("duffing", "physics", t_end=0.01)

        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "duffing"
