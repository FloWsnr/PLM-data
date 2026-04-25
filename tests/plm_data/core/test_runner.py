"""Tests for plm_data.core.runner."""

from dataclasses import replace
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import plm_data.core.runner as runner_mod
from plm_data.core.runner import SimulationRunner


class _FakeComm:
    def __init__(self, rank: int, broadcast_value=None):
        self.rank = rank
        self._broadcast_value = broadcast_value

    def bcast(self, value, root=0):
        if self.rank == root:
            return value
        return self._broadcast_value


def test_simulation_runner_heat(heat_config):
    runner = SimulationRunner(heat_config)
    summary = runner.run(console_level=logging.WARNING)

    assert summary["pde"] == "heat"
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
    assert meta["pde"] == "heat"
    assert meta["domain_type"] == "rectangle"
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
    assert run_meta["config"]["resolved"]["pde"] == "heat"
    assert run_meta["config"]["resolved"]["seed"] == 42


def test_simulation_runner_cleans_output_dir_before_single_run(heat_config):
    output_dir = Path(heat_config.output.path) / "basic" / "heat"
    stale_dir = output_dir / "stale_dir"
    stale_dir.mkdir(parents=True)
    (output_dir / "stale.npy").write_text("stale")
    (stale_dir / "stale.txt").write_text("stale")

    runner = SimulationRunner(heat_config)
    summary = runner.run(console_level=logging.WARNING)

    assert Path(summary["output_dir"]) == output_dir
    assert not (output_dir / "stale.npy").exists()
    assert not stale_dir.exists()
    assert (output_dir / "u.npy").is_file()


def test_simulation_runner_requires_explicit_seed(heat_config):
    with pytest.raises(ValueError, match="explicit seed"):
        SimulationRunner(replace(heat_config, seed=None))


def test_prepare_output_dir_only_resets_on_rank_zero(monkeypatch, tmp_path):
    calls: list[Path] = []

    def fake_reset(path: Path) -> None:
        calls.append(path)

    monkeypatch.setattr(runner_mod, "_reset_output_dir", fake_reset)

    runner_mod._prepare_output_dir(
        tmp_path / "out",
        reset=True,
        comm=_FakeComm(rank=1, broadcast_value=None),
    )

    assert calls == []


def test_prepare_output_dir_broadcasts_rank_zero_failure(monkeypatch, tmp_path):
    def fake_reset(path: Path) -> None:
        raise FileNotFoundError("synthetic cleanup failure")

    monkeypatch.setattr(runner_mod, "_reset_output_dir", fake_reset)

    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        runner_mod._prepare_output_dir(
            tmp_path / "out",
            reset=True,
            comm=_FakeComm(rank=0),
        )


def test_cleanup_runtime_problem_destroys_unique_petsc_handles_once():
    class Handle:
        def __init__(self) -> None:
            self.destroy_calls = 0

        def destroy(self) -> None:
            self.destroy_calls += 1

    class FakeSolver:
        def getConvergedReason(self) -> int:
            return 1

    shared_solver = Handle()
    shared_matrix = Handle()
    rhs = Handle()
    solution = Handle()
    preconditioner = Handle()

    wrapper_a = SimpleNamespace(
        _solver=shared_solver,
        _A=shared_matrix,
        _b=rhs,
        _x=solution,
        _P_mat=preconditioner,
        solver=FakeSolver(),
    )
    wrapper_b = SimpleNamespace(
        _solver=shared_solver,
        _A=shared_matrix,
        _b=rhs,
        _x=solution,
        _P_mat=preconditioner,
        solver=FakeSolver(),
    )
    runtime_problem = SimpleNamespace(
        primary=wrapper_a,
        nested={"secondary": wrapper_b},
    )

    runner_mod._cleanup_runtime_problem(runtime_problem)

    for handle in (
        shared_solver,
        shared_matrix,
        rhs,
        solution,
        preconditioner,
    ):
        assert handle.destroy_calls == 1
    assert wrapper_a._solver is None
    assert wrapper_a._A is None
    assert wrapper_a._b is None
    assert wrapper_a._x is None
    assert wrapper_a._P_mat is None
    assert wrapper_b._solver is None
    assert wrapper_b._A is None
    assert wrapper_b._b is None
    assert wrapper_b._x is None
    assert wrapper_b._P_mat is None


def test_run_random_simulation_retries_and_cleans_failed_attempt(monkeypatch, tmp_path):
    class Sampled:
        def __init__(self, attempt: int) -> None:
            self.config = SimpleNamespace(seed=99)
            self.pde_name = "heat"
            self.domain_name = "rectangle"
            self.attempt = attempt

        def output_dir(self, output_root):
            return Path(output_root) / f"attempt_{self.attempt}"

        def metadata(self):
            return {"run_id": f"run_{self.attempt}", "attempt": self.attempt}

    samples = [Sampled(0), Sampled(1)]
    cleaned: list[Path] = []
    runs: list[Path] = []

    def fake_sample(seed: int, attempt: int):
        assert seed == 99
        return samples[attempt]

    class FakeRunner:
        def __init__(
            self,
            config,
            output_root,
            *,
            output_dir_override,
            sampled_info,
        ) -> None:
            self.output_dir_override = Path(output_dir_override)
            runs.append(self.output_dir_override)

        def run(self, console_level):
            if len(runs) == 1:
                raise RuntimeError("synthetic failure")
            return {"output_dir": str(self.output_dir_override)}

    monkeypatch.setattr(runner_mod, "_sample_random_config_on_rank_zero", fake_sample)
    monkeypatch.setattr(runner_mod, "SimulationRunner", FakeRunner)
    monkeypatch.setattr(
        runner_mod,
        "_cleanup_failed_random_output",
        lambda output_dir: cleaned.append(Path(output_dir)),
    )

    summary = runner_mod.run_random_simulation(
        seed=99,
        output_root=tmp_path,
        attempt_budget=2,
    )

    assert cleaned == [tmp_path / "attempt_0"]
    assert summary["output_dir"] == str(tmp_path / "attempt_1")
    random_sampling = summary["random_sampling"]
    assert isinstance(random_sampling, dict)
    assert random_sampling["run_id"] == "run_1"


def test_simulation_runner_writes_failure_metadata(heat_config, monkeypatch):
    runner = SimulationRunner(heat_config)

    class _FailingProblem:
        def run(self, output):
            raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(
        runner.pde,
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
    assert run_meta["config"]["resolved"]["pde"] == "heat"
