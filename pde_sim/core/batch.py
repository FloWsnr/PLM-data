"""Batch execution utilities (shared by CLI and scripts)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from pde_sim.core.logging import restore_stdout, setup_logging
from pde_sim.core.simulation import run_from_config


def collect_yaml_files(config_dir: Path, pattern: str = "*.yaml") -> list[Path]:
    """Collect YAML config files from a directory (supports .yaml and .yml)."""
    yaml_files = list(config_dir.glob(pattern))
    if pattern.endswith(".yaml"):
        yaml_files.extend(config_dir.glob(pattern[:-5] + ".yml"))
    return sorted(set(yaml_files))


def run_batch(
    *,
    config_dir: Path,
    start_index: int = 1,
    log_file: Path | None = None,
    quiet: bool = False,
    pattern: str = "*.yaml",
    output_dir: Path | None = None,
    seed: int | None = None,
    storage: str | None = None,
    keep_storage: bool | None = None,
    unique_suffix: bool | None = True,
    overwrite: bool = False,
    continue_on_error: bool = True,
) -> tuple[int, int]:
    """Run a batch of simulations from a directory of YAML configs."""
    logger = setup_logging(
        log_file=log_file,
        console=not quiet,
        capture_stdout=log_file is not None,
    )

    if not config_dir.is_dir():
        logger.error("Config dir is not a directory: %s", config_dir)
        return 0, 0

    config_files = collect_yaml_files(config_dir, pattern)
    if not config_files:
        logger.error("No YAML config files found in %s", config_dir)
        return 0, 0

    total = len(config_files)
    logger.info("Found %d config files in %s", total, config_dir)
    logger.info("Starting from index %d", start_index)
    if output_dir is not None:
        logger.info("Overriding output dir: %s", output_dir)
    if storage is not None:
        logger.info("Overriding output storage: %s", storage)
    if log_file is not None:
        logger.info("Logging to: %s", log_file)

    ok = 0
    failed = 0

    for i, config_path in enumerate(config_files, start=1):
        if i < start_index:
            continue

        logger.info("=" * 60)
        logger.info("Simulation %d/%d: %s", i, total, config_path.name)

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            logger.info("Preset: %s", cfg.get("preset", "unknown"))
            logger.info("Parameters: %s", cfg.get("parameters", {}))

            metadata = run_from_config(
                config_path=config_path,
                output_dir=output_dir,
                seed=seed,
                verbose=True,
                overwrite=overwrite,
                storage=storage,
                keep_storage=keep_storage,
                unique_suffix=unique_suffix,
            )
            logger.info("Output: %s", metadata.get("folder_name"))
            ok += 1
        except Exception as e:
            logger.error("Simulation %d (%s) failed: %s", i, config_path.name, e)
            failed += 1
            if not continue_on_error:
                break

    logger.info("=" * 60)
    logger.info("Batch complete: %d successful, %d failed", ok, failed)
    logger.info("=" * 60)

    restore_stdout()

    # Provide a non-zero exit for scripts calling into this
    if failed and not continue_on_error:
        sys.stderr.write("Batch stopped due to failure\n")

    return ok, failed

