"""Tests for running actual PDE preset simulations."""

import numpy as np

from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset


def test_heat_preset_single_step(tmp_path, heat_config):
    preset = get_preset("heat")
    output_dir = tmp_path / "basic" / "heat"
    writer = FrameWriter(output_dir, heat_config)

    result = preset.run(heat_config, writer)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert result.num_dofs > 0

    # Initial frame + 1 timestep frame = 2 frames
    assert writer.frame_count == 2
    writer.finalize()
    npy_path = output_dir / "u.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (2, *heat_config.output_resolution)


def test_poisson_preset(tmp_path, poisson_config):
    preset = get_preset("poisson")
    output_dir = tmp_path / "basic" / "poisson"
    writer = FrameWriter(output_dir, poisson_config)

    result = preset.run(poisson_config, writer)

    assert result.solver_converged is True
    assert result.num_dofs > 0
    assert writer.frame_count == 1
    writer.finalize()
    npy_path = output_dir / "u.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (1, *poisson_config.output_resolution)
    # Poisson with sin source on unit square should have positive interior values
    assert np.max(arr) > 0
