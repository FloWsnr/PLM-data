"""Tests for GIF gallery generation."""

from pathlib import Path

import pytest

from plm_data.tools.gif_gallery import (
    collect_field_names,
    collect_gif_runs,
    write_gallery_html,
)


def _write_fake_gif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GIF89a")


def test_collect_gif_runs_groups_fields_by_directory(tmp_path):
    _write_fake_gif(tmp_path / "basic" / "heat" / "u.gif")
    _write_fake_gif(tmp_path / "fluids" / "navier_stokes" / "pressure.gif")
    _write_fake_gif(tmp_path / "fluids" / "navier_stokes" / "velocity_x.gif")
    _write_fake_gif(tmp_path / "fluids" / "navier_stokes" / "velocity_y.gif")

    runs = collect_gif_runs(tmp_path)

    assert [run.label for run in runs] == ["basic/heat", "fluids/navier_stokes"]
    assert collect_field_names(runs) == ["pressure", "u", "velocity_x", "velocity_y"]
    assert sorted(runs[0].fields) == ["u"]
    assert sorted(runs[1].fields) == ["pressure", "velocity_x", "velocity_y"]


def test_write_gallery_html_emits_relative_gif_paths(tmp_path):
    _write_fake_gif(tmp_path / "basic" / "heat" / "u.gif")
    _write_fake_gif(tmp_path / "fluids" / "navier_stokes" / "pressure.gif")
    _write_fake_gif(tmp_path / "fluids" / "navier_stokes" / "velocity_x.gif")

    summary = write_gallery_html(tmp_path, title="Test Gallery")
    html = summary.output_path.read_text(encoding="utf-8")

    assert summary.num_rows == 2
    assert summary.num_fields == 3
    assert "<h1>Test Gallery</h1>" in html
    assert ">basic/heat<" in html
    assert ">fluids/navier_stokes<" in html
    assert 'src="basic/heat/u.gif"' in html
    assert 'src="fluids/navier_stokes/pressure.gif"' in html
    assert 'src="fluids/navier_stokes/velocity_x.gif"' in html
    assert '<td class="empty"></td>' in html


def test_write_gallery_html_requires_gifs(tmp_path):
    with pytest.raises(ValueError, match="No GIF files found"):
        write_gallery_html(tmp_path)


def test_collect_gif_runs_uses_run_meta_for_root_directory_label(tmp_path):
    _write_fake_gif(tmp_path / "u.gif")
    (tmp_path / "run_meta.json").write_text(
        '{"category": "basic", "preset": "heat"}',
        encoding="utf-8",
    )

    runs = collect_gif_runs(tmp_path)

    assert [run.label for run in runs] == ["basic/heat"]
