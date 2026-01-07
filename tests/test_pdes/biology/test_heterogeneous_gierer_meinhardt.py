"""Tests for Heterogeneous Gierer-Meinhardt PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from tests.conftest import run_short_simulation


class TestHeterogeneousGiererMeinhardtPDE:
    """Tests for Heterogeneous Gierer-Meinhardt PDE."""

    def test_registered(self):
        """Test that heterogeneous-gierer-meinhardt is registered."""
        assert "heterogeneous-gierer-meinhardt" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("heterogeneous-gierer-meinhardt")
        meta = preset.metadata

        assert meta.name == "heterogeneous-gierer-meinhardt"
        assert meta.category == "biology"
        assert meta.num_fields == 2

    def test_short_simulation(self):
        """Test running a short simulation using default config."""
        result, config = run_short_simulation("heterogeneous-gierer-meinhardt", "biology", t_end=0.01)

        # Check result type and finite values
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert np.isfinite(result[0].data).all()
        assert config["preset"] == "heterogeneous-gierer-meinhardt"
