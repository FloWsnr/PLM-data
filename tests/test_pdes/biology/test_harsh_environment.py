"""Tests for Allee effect model."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestHarshEnvironmentPDE:
    """Tests for Allee effect model."""

    def test_registered(self):
        """Test that harsh-environment is registered."""
        assert "harsh-environment" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("harsh-environment")
        meta = preset.metadata

        assert meta.name == "harsh-environment"
        assert meta.category == "biology"
        assert meta.num_fields == 1

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("harsh-environment", "biology", t_end=0.01)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "harsh-environment"
