"""Tests for Hyperbolic Brusselator PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestHyperbolicBrusselatorPDE:
    """Tests for Hyperbolic Brusselator PDE."""

    def test_registered(self):
        """Test that hyperbolic-brusselator is registered."""
        assert "hyperbolic-brusselator" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("hyperbolic-brusselator")
        meta = preset.metadata

        assert meta.name == "hyperbolic-brusselator"
        assert meta.category == "biology"
        assert meta.num_fields == 4
        assert "u" in meta.field_names
        assert "v" in meta.field_names
        assert "w" in meta.field_names
        assert "q" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("hyperbolic-brusselator", "biology", t_end=0.001)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "hyperbolic-brusselator"
