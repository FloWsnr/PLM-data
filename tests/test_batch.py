"""Tests for batch config discovery helpers."""

from pde_sim.core.batch import collect_yaml_files


def test_collect_yaml_files_recursive_default_excludes_master(tmp_path):
    """Default collection is recursive and excludes master config files."""
    (tmp_path / "master.yaml").write_text("seed: 123\n")
    (tmp_path / "top.yaml").write_text("preset: heat\n")
    (tmp_path / "top.yml").write_text("preset: wave\n")

    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    (nested / "child.yaml").write_text("preset: advection\n")
    (nested / "master.yml").write_text("seed: 999\n")

    files = collect_yaml_files(tmp_path)
    rel = [str(p.relative_to(tmp_path)) for p in files]

    assert rel == [
        "nested/deeper/child.yaml",
        "top.yaml",
        "top.yml",
    ]

