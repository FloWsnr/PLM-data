"""Tests for Bacteria Advection PDE."""

import numpy as np
import pytest

from pde import ScalarField

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestBacteriaAdvectionPDE:
    """Tests for Bacteria Advection PDE."""

    def test_registered(self):
        """Test that bacteria-advection is registered."""
        assert "bacteria-advection" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("bacteria-advection")
        meta = preset.metadata

        assert meta.name == "bacteria-advection"
        assert meta.category == "biology"
        assert meta.num_fields == 1
        assert "C" in meta.field_names

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("bacteria-advection", "biology", t_end=0.1)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, ScalarField)
        assert np.isfinite(result.data).all()
        assert config["preset"] == "bacteria-advection"
