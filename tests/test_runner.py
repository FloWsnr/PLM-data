"""Tests for plm_data.core.runner."""

from plm_data.core.runner import SimulationRunner


def test_simulation_runner_heat(heat_config):
    runner = SimulationRunner(heat_config)
    summary = runner.run(verbose=False)

    assert summary["preset"] == "heat"
    assert summary["category"] == "basic"
    assert summary["solver_converged"] is True
    assert summary["num_frames"] == 2
    assert summary["num_dofs"] > 0
    assert summary["wall_time"] > 0
