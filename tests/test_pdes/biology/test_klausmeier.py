"""Tests for Klausmeier vegetation-water PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestKlausmeierPDE:
    """Tests for Klausmeier vegetation-water PDE."""

    def test_registered(self):
        """Test that klausmeier is registered."""
        assert "klausmeier" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("klausmeier")
        meta = preset.metadata

        assert meta.name == "klausmeier"
        assert meta.category == "biology"
        assert meta.num_fields == 2
        assert "n" in meta.field_names
        assert "w" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("klausmeier", "biology", t_end=0.1)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "klausmeier"
