"""Tests for Zakharov-Kuznetsov PDE (2D version of KdV)."""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField

from pde_sim.pdes import get_pde_preset, list_presets

from tests.conftest import run_short_simulation


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


class TestZakharovKuznetsovPDE:
    """Tests for Zakharov-Kuznetsov PDE."""

    def test_registered(self):
        """Test that zakharov-kuznetsov is registered."""
        assert "zakharov-kuznetsov" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("zakharov-kuznetsov")
        meta = preset.metadata

        assert meta.name == "zakharov-kuznetsov"
        assert meta.category == "physics"
        assert meta.num_fields == 1
        assert "u" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("zakharov-kuznetsov", "physics", t_end=0.0001)

        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "zakharov-kuznetsov"
