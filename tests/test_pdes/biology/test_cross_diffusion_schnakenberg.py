"""Tests for Cross-Diffusion Schnakenberg PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestCrossDiffusionSchnakenbergPDE:
    """Tests for Cross-Diffusion Schnakenberg PDE."""

    def test_registered(self):
        """Test that cross-diffusion-schnakenberg is registered."""
        assert "cross-diffusion-schnakenberg" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("cross-diffusion-schnakenberg")
        meta = preset.metadata

        assert meta.name == "cross-diffusion-schnakenberg"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "u" in meta.field_names
        assert "v" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("cross-diffusion-schnakenberg", "biology", t_end=0.1)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert config["preset"] == "cross-diffusion-schnakenberg"
