"""Tests for plm_data.core.runner."""

from dataclasses import replace
import logging

import pytest

from plm_data.core.runner import SimulationRunner


def test_simulation_runner_heat(heat_config):
    runner = SimulationRunner(heat_config)
    summary = runner.run(console_level=logging.WARNING)

    assert summary["preset"] == "heat"
    assert summary["category"] == "basic"
    assert summary["solver_converged"] is True
    assert summary["num_frames"] == 2
    assert summary["num_dofs"] > 0
    assert summary["wall_time"] > 0
    assert summary["timings"]["total_wall_seconds"] == summary["wall_time"]
    assert summary["timings"]["problem_run_seconds"] > 0
    assert summary["timings"]["output_finalize_call_seconds"] >= 0
    assert summary["timings"]["output"]["frame_count"] == 2


def test_simulation_runner_requires_explicit_seed(heat_config):
    with pytest.raises(ValueError, match="explicit seed"):
        SimulationRunner(replace(heat_config, seed=None))
