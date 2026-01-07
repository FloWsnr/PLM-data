"""Tests for Perona-Malik edge-preserving diffusion PDE."""

import numpy as np
import pytest

from pde import CartesianGrid

from pde_sim.pdes import get_pde_preset, list_presets

from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 1], [0, 1]], [16, 16], periodic=True)


class TestPeronaMalikPDE:
    """Tests for Perona-Malik edge-preserving diffusion."""

    def test_registered(self):
        """Test that perona-malik is registered."""
        assert "perona-malik" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("perona-malik")
        meta = preset.metadata

        assert meta.name == "perona-malik"
        assert meta.category == "physics"
        assert meta.num_fields == 1

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("perona-malik", "physics", t_end=0.001)

        assert result is not None
        assert np.isfinite(result.data).all()
        assert config["preset"] == "perona-malik"
