"""Tests for plm_data.core.runner."""

from dataclasses import replace
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def test_simulation_runner_materializes_sampled_runtime_values(tmp_path):
    config_path = tmp_path / "heat_sampled_runtime.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "preset": "heat",
                "parameters": {},
                "coefficients": {
                    "kappa": {
                        "type": "constant",
                        "params": {
                            "value": {"sample": "uniform", "min": 0.8, "max": 1.2}
                        },
                    }
                },
                "domain": {
                    "type": "rectangle",
                    "size": [1.0, 1.0],
                    "mesh_resolution": [6, 6],
                },
                "inputs": {
                    "u": {
                        "source": {
                            "type": "constant",
                            "params": {"value": 1.0},
                        },
                        "initial_condition": {
                            "type": "constant",
                            "params": {"value": 0.0},
                        },
                    }
                },
                "boundary_conditions": {
                    "u": {
                        side: [
                            {
                                "operator": "dirichlet",
                                "value": {
                                    "type": "constant",
                                    "params": {
                                        "value": {
                                            "sample": "uniform",
                                            "min": -0.2,
                                            "max": 0.2,
                                        }
                                    },
                                },
                            }
                        ]
                        for side in ("x-", "x+", "y-", "y+")
                    }
                },
                "output": {
                    "resolution": [8, 8],
                    "num_frames": 1,
                    "formats": ["numpy"],
                    "fields": {"u": "scalar"},
                },
                "solver": {
                    "strategy": "constant_lhs_scalar_spd",
                    "serial": {"ksp_type": "preonly", "pc_type": "lu"},
                    "mpi": {"ksp_type": "preonly", "pc_type": "lu"},
                },
                "time": {"dt": 0.01, "t_end": 0.01},
                "seed": 42,
            }
        )
    )

    runner_a = SimulationRunner.from_yaml(config_path, tmp_path / "out_a", seed=7)
    runner_b = SimulationRunner.from_yaml(config_path, tmp_path / "out_b", seed=7)
    runner_c = SimulationRunner.from_yaml(config_path, tmp_path / "out_c", seed=8)

    bc_a = runner_a.config.boundary_field("u").side_conditions("x-")[0]
    bc_b = runner_b.config.boundary_field("u").side_conditions("x-")[0]
    bc_c = runner_c.config.boundary_field("u").side_conditions("x-")[0]

    assert bc_a.value is not None
    assert bc_b.value is not None
    assert bc_c.value is not None
    kappa_a = runner_a.config.coefficient("kappa").params["value"]
    kappa_b = runner_b.config.coefficient("kappa").params["value"]
    kappa_c = runner_c.config.coefficient("kappa").params["value"]

    assert kappa_a == kappa_b
    assert bc_a.value.params["value"] == bc_b.value.params["value"]
    assert (
        kappa_a != kappa_c or bc_a.value.params["value"] != bc_c.value.params["value"]
    )


def test_simulation_runner_run_many_from_yaml_uses_seed_subdirs(tmp_path):
    config_path = tmp_path / "heat_batch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "preset": "heat",
                "parameters": {},
                "domain": {
                    "type": "rectangle",
                    "size": [1.0, 1.0],
                    "mesh_resolution": [8, 8],
                },
                "coefficients": {
                    "kappa": {"type": "constant", "params": {"value": 0.01}}
                },
                "inputs": {
                    "u": {
                        "source": {"type": "none"},
                        "initial_condition": {
                            "type": "gaussian_bump",
                            "params": {
                                "sigma": 0.1,
                                "amplitude": 1.0,
                                "center": [0.5, 0.5],
                            },
                        },
                    }
                },
                "boundary_conditions": {
                    "u": {
                        side: [
                            {
                                "operator": "neumann",
                                "value": {
                                    "type": "constant",
                                    "params": {"value": 0.0},
                                },
                            }
                        ]
                        for side in ("x-", "x+", "y-", "y+")
                    }
                },
                "output": {
                    "resolution": [4, 4],
                    "num_frames": 2,
                    "formats": ["numpy"],
                    "fields": {"u": "scalar"},
                },
                "solver": {
                    "strategy": "constant_lhs_scalar_spd",
                    "serial": {"ksp_type": "preonly", "pc_type": "lu"},
                    "mpi": {"ksp_type": "preonly", "pc_type": "lu"},
                },
                "time": {"dt": 0.01, "t_end": 0.01},
                "seed": 42,
            }
        )
    )

    batch_output_dir = tmp_path / "out" / "basic" / "heat"
    batch_output_dir.mkdir(parents=True)
    (batch_output_dir / "stale.txt").write_text("stale")

    summaries = SimulationRunner.run_many_from_yaml(
        config_path,
        tmp_path / "out",
        n_runs=2,
        console_level=logging.WARNING,
    )

    assert [Path(summary["output_dir"]).name for summary in summaries] == [
        "seed_42",
        "seed_43",
    ]
    assert not (batch_output_dir / "stale.txt").exists()

    for expected_seed, summary in zip((42, 43), summaries, strict=True):
        run_dir = Path(summary["output_dir"])
        assert run_dir == batch_output_dir / f"seed_{expected_seed}"
        assert (run_dir / "u.npy").is_file()
        run_meta = json.loads((run_dir / "run_meta.json").read_text())
        assert run_meta["config"]["resolved"]["seed"] == expected_seed


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
