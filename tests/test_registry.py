"""Tests for plm_data.presets registry."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from plm_data.presets import get_preset, list_presets
from plm_data.presets.base import PDEPreset


def test_list_presets():
    presets = list_presets()
    assert "advection" in presets
    assert "bistable_travelling_waves" in presets
    assert "burgers" in presets
    assert "compressible_euler" in presets
    assert "elasticity" in presets
    assert "fisher_kpp" in presets
    assert "shallow_water" in presets
    assert "heat" in presets
    assert "plate" in presets
    assert "schrodinger" in presets
    assert "wave" in presets
    assert "poisson" in presets
    assert "cahn_hilliard" in presets
    assert "cgl" in presets
    assert "darcy" in presets
    assert "van_der_pol" in presets
    assert "maxwell" in presets
    assert "maxwell_pulse" in presets
    assert "navier_stokes" in presets
    assert "stokes" in presets
    assert "thermal_convection" in presets
    assert "cyclic_competition" in presets


def test_get_preset():
    preset = get_preset("heat")
    assert isinstance(preset, PDEPreset)


def test_get_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("nonexistent_preset")


def test_registry_import_does_not_create_pyclaw_log(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath = str(repo_root)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath = f"{pythonpath}:{existing_pythonpath}"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from plm_data.presets import list_presets; "
                "print('compressible_euler' in list_presets())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )

    assert result.stdout.strip() == "True"
    assert not (tmp_path / "pyclaw.log").exists()
