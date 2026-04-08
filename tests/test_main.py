"""Tests for the CLI entry point."""

from types import SimpleNamespace

import pytest

import plm_data.__main__ as main_mod
import plm_data.core.runner as runner_mod
import plm_data.tools.gif_gallery as gallery_mod


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


def test_cmd_gallery_passes_arguments(monkeypatch, tmp_path, capsys):
    calls: dict[str, object] = {}

    class Summary:
        output_path = tmp_path / "gallery.html"
        num_rows = 3
        num_fields = 4

    def fake_write_gallery_html(directory, output_path, *, title):
        calls["directory"] = directory
        calls["output_path"] = output_path
        calls["title"] = title
        return Summary()

    monkeypatch.setattr(gallery_mod, "write_gallery_html", fake_write_gallery_html)

    main_mod.cmd_gallery(
        SimpleNamespace(
            directory=str(tmp_path),
            output=str(tmp_path / "custom.html"),
            title="Custom Title",
        )
    )

    assert calls["directory"] == str(tmp_path)
    assert calls["output_path"] == str(tmp_path / "custom.html")
    assert calls["title"] == "Custom Title"
    assert "3 PDE rows, 4 fields" in capsys.readouterr().out
