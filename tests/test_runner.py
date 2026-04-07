"""Tests for plm_data.core.runner."""

from dataclasses import replace
import json
import logging
from pathlib import Path

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

    meta = json.loads((Path(summary["output_dir"]) / "frames_meta.json").read_text())
    run_meta = json.loads((Path(summary["output_dir"]) / "run_meta.json").read_text())
    assert meta["diagnostics"]["solver_health"]["applied"] is True
    assert meta["diagnostics"]["solver_health"]["status"] == "pass"
    assert meta["diagnostics"]["runtime_health"]["applied"] is False
    assert meta["diagnostics"]["overall_health"]["status"] == "pass"
    assert run_meta["status"] == "success"
    assert run_meta["stage"] == "completed"
    assert run_meta["error"] is None
    assert run_meta["frames_meta_path"] == str(
        Path(summary["output_dir"]) / "frames_meta.json"
    )
    assert run_meta["num_frames"] == summary["num_frames"]
    assert run_meta["summary"]["health_status"] == "pass"
    assert run_meta["summary"]["status"] == "success"
    assert run_meta["config"]["source_path"] is None
    assert run_meta["config"]["resolved"]["preset"] == "heat"
    assert run_meta["config"]["resolved"]["seed"] == 42


def test_simulation_runner_requires_explicit_seed(heat_config):
    with pytest.raises(ValueError, match="explicit seed"):
        SimulationRunner(replace(heat_config, seed=None))


def test_simulation_runner_writes_failure_metadata(heat_config, monkeypatch):
    runner = SimulationRunner(heat_config)

    class _FailingProblem:
        def run(self, output):
            raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(
        runner.preset,
        "build_problem",
        lambda config: _FailingProblem(),
    )

    with pytest.raises(RuntimeError, match="synthetic runner failure"):
        runner.run(console_level=logging.WARNING)

    output_dir = Path(heat_config.output.path) / "basic" / "heat"
    run_meta = json.loads((output_dir / "run_meta.json").read_text())
    assert run_meta["status"] == "failed"
    assert run_meta["stage"] == "problem_run"
    assert run_meta["error"]["stage"] == "problem_run"
    assert run_meta["error"]["type"] == "RuntimeError"
    assert run_meta["error"]["message"] == "synthetic runner failure"
    assert run_meta["frames_meta_path"] is None
    assert run_meta["num_frames"] == 0
    assert run_meta["timings"]["output"]["frame_count"] == 0
    assert run_meta["summary"]["health_status"] == "fail"
    assert run_meta["summary"]["status"] == "failed"
    assert run_meta["config"]["resolved"]["preset"] == "heat"


def test_simulation_runner_compressible_euler_preserves_logging(tmp_path):
    runner = SimulationRunner.from_yaml(
        "configs/fluids/compressible_euler/2d_reflective_chaotic_quadrant_smoke.yaml",
        tmp_path,
    )

    summary = runner.run(console_level=logging.WARNING)

    assert summary["preset"] == "compressible_euler"
    assert summary["solver_converged"] is True
    log_text = (Path(summary["output_dir"]) / "simulation.log").read_text()
    assert "Setup: output sampling" in log_text
    assert "Time stepping complete" in log_text
