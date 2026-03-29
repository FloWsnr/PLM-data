"""Tests for plm_data.presets registry."""

import pytest

from plm_data.presets import get_preset, list_presets
from plm_data.presets.base import PDEPreset


def test_list_presets():
    presets = list_presets()
    assert "heat" in presets
    assert "plate" in presets
    assert "wave" in presets
    assert "poisson" in presets
    assert "cahn_hilliard" in presets
    assert "maxwell" in presets
    assert "maxwell_pulse" in presets
    assert "navier_stokes" in presets
    assert "stokes" in presets
    assert "thermal_convection" in presets


def test_get_preset():
    preset = get_preset("heat")
    assert isinstance(preset, PDEPreset)


def test_get_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("nonexistent_preset")
