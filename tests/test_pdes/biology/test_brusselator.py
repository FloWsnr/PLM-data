"""Tests for Brusselator PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestBrusselatorPDE:
    """Tests for Brusselator PDE."""

    def test_registered(self):
        """Test that brusselator is registered."""
        assert "brusselator" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("brusselator")
        meta = preset.metadata

        assert meta.name == "brusselator"
        assert meta.category == "biology"

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("brusselator", "biology", t_end=0.001)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "brusselator"
