"""Tests for the CLI entry point."""

from types import SimpleNamespace

import pytest

import plm_data.__main__ as main_mod
import plm_data.core.runner as runner_mod


def test_cmd_random_passes_seed_and_output_dir(monkeypatch, tmp_path, capsys):
    calls: dict[str, object] = {}

    class FakeComm:
        rank = 0

    class FakeMPI:
        COMM_WORLD = FakeComm()

    def fake_run_random_simulation(*, seed, output_root, console_level):
        calls["seed"] = seed
        calls["output_root"] = output_root
        calls["console_level"] = console_level
        return {"output_dir": str(tmp_path / "out")}

    monkeypatch.setattr(runner_mod, "run_random_simulation", fake_run_random_simulation)
    monkeypatch.setattr("mpi4py.MPI", FakeMPI)

    summary = main_mod.cmd_random(
        SimpleNamespace(seed=1234, output_dir=str(tmp_path), log_level="WARNING")
    )

    assert calls["seed"] == 1234
    assert calls["output_root"] == str(tmp_path)
    assert summary["output_dir"] == str(tmp_path / "out")
    assert str(tmp_path / "out") in capsys.readouterr().out


def test_main_requires_seed(monkeypatch):
    monkeypatch.setattr("sys.argv", ["plm_data"])

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code == 2


def test_main_rejects_removed_run_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["plm_data", "run", "old-entrypoint"])

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code == 2


def test_main_rejects_removed_gallery_subcommand(monkeypatch):
    monkeypatch.setattr("sys.argv", ["plm_data", "gallery", "./output"])

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code == 2
