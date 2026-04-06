"""Tests for the CLI entry point."""

from types import SimpleNamespace

import pytest

import plm_data.__main__ as main_mod
import plm_data.core.runner as runner_mod


def test_cmd_run_passes_output_dir(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class DummyRunner:
        def run(self, console_level):
            calls["console_level"] = console_level

    class FakeSimulationRunner:
        @classmethod
        def from_yaml(cls, config_path, output_dir, *, seed=None):
            calls["config_path"] = config_path
            calls["output_dir"] = output_dir
            calls["seed"] = seed
            return DummyRunner()

    monkeypatch.setattr(runner_mod, "SimulationRunner", FakeSimulationRunner)

    main_mod.cmd_run(
        SimpleNamespace(
            config="configs/basic/heat/2d_localized_blob_diffusion.yaml",
            output_dir=str(tmp_path),
            log_level="WARNING",
            seed=None,
        )
    )

    assert calls["config_path"] == "configs/basic/heat/2d_localized_blob_diffusion.yaml"
    assert calls["output_dir"] == str(tmp_path)
    assert calls["seed"] is None


def test_cmd_run_passes_seed_override(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class DummyRunner:
        def run(self, console_level):
            calls["console_level"] = console_level

    class FakeSimulationRunner:
        @classmethod
        def from_yaml(cls, config_path, output_dir, *, seed=None):
            calls["config_path"] = config_path
            calls["output_dir"] = output_dir
            calls["seed"] = seed
            return DummyRunner()

    monkeypatch.setattr(runner_mod, "SimulationRunner", FakeSimulationRunner)

    main_mod.cmd_run(
        SimpleNamespace(
            config="configs/basic/heat/2d_localized_blob_diffusion.yaml",
            output_dir=str(tmp_path),
            log_level="INFO",
            seed=123,
        )
    )

    assert calls["config_path"] == "configs/basic/heat/2d_localized_blob_diffusion.yaml"
    assert calls["output_dir"] == str(tmp_path)
    assert calls["seed"] == 123


def test_main_run_requires_output_dir(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["plm_data", "run", "configs/basic/heat/2d_localized_blob_diffusion.yaml"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code == 2
