"""Tests for tumor-immune interaction model."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestImmunotherapyPDE:
    """Tests for tumor-immune interaction model."""

    def test_registered(self):
        """Test that immunotherapy is registered."""
        assert "immunotherapy" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("immunotherapy")
        meta = preset.metadata

        assert meta.name == "immunotherapy"
        assert meta.category == "biology"
        assert meta.num_fields == 3
        assert "u" in meta.field_names  # effector cells
        assert "v" in meta.field_names  # tumor
        assert "w" in meta.field_names  # cytokine

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("immunotherapy", "biology", t_end=0.01)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert np.isfinite(result[1].data).all()
        assert np.isfinite(result[2].data).all()
        assert config["preset"] == "immunotherapy"
