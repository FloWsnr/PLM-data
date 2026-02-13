"""Tests for batch config discovery helpers and parallel execution."""

import json

import yaml

from pde_sim.core.batch import collect_yaml_files, run_batch


_MINIMAL_CONFIG = {
    "preset": "heat",
    "parameters": {"D_T": 0.01},
    "init": {"type": "random-uniform", "params": {"low": 0.0, "high": 1.0}},
    "solver": "euler",
    "backend": "numpy",
    "t_end": 0.005,
    "dt": 0.0001,
    "resolution": [16, 16],
    "bc": {"x-": "periodic", "x+": "periodic", "y-": "periodic", "y+": "periodic"},
    "seed": 42,
}


def _write_config(path, **overrides):
    """Write a minimal heat config with optional overrides."""
    cfg = {**_MINIMAL_CONFIG, **overrides}
    cfg["output"] = {"path": str(path.parent), "num_frames": 3, "formats": ["png"]}
    with open(path, "w") as f:
        yaml.dump(cfg, f)


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


def test_run_batch_parallel(tmp_path):
    """Parallel batch produces the same outputs as sequential."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create 3 minimal configs with different seeds for unique output
    for i in range(3):
        _write_config(config_dir / f"run_{i}.yaml", seed=100 + i)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    ok, failed = run_batch(
        config_dir=config_dir,
        output_dir=output_dir,
        num_processes=2,
        unique_suffix=False,
    )

    assert ok == 3
    assert failed == 0

    # Each config should have produced an output folder with metadata
    metadata_files = list(output_dir.rglob("metadata.json"))
    assert len(metadata_files) == 3

    for mf in metadata_files:
        with open(mf) as f:
            meta = json.load(f)
        assert meta["preset"] == "heat"
        assert meta["simulation"]["totalFrames"] == 3


def test_run_batch_parallel_log_file_creates_main_and_per_sim_logs(tmp_path):
    """Batch logging writes one main file and one per-simulation file."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    for i in range(2):
        _write_config(config_dir / f"run_{i}.yaml", seed=200 + i)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    main_log = tmp_path / "logs" / "batch.log"

    ok, failed = run_batch(
        config_dir=config_dir,
        output_dir=output_dir,
        num_processes=2,
        unique_suffix=False,
        log_file=main_log,
        quiet=True,
    )

    assert ok == 2
    assert failed == 0
    assert main_log.exists()

    main_log_text = main_log.read_text()
    assert "Batch complete" in main_log_text

    simulation_logs = sorted(main_log.parent.glob(f"{main_log.stem}_*.log"))
    assert len(simulation_logs) == 2

    for sim_log in simulation_logs:
        text = sim_log.read_text()
        assert "Starting simulation:" in text
