"""Tests for Fisher-KPP PDE."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestFisherKPPPDE:
    """Tests for Fisher-KPP PDE."""

    def test_registered(self):
        """Test that fisher-kpp is registered."""
        assert "fisher-kpp" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("fisher-kpp")
        meta = preset.metadata

        assert meta.name == "fisher-kpp"
        assert meta.category == "biology"

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("fisher-kpp", "biology", t_end=0.01)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "fisher-kpp"
