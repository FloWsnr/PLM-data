"""Batch execution utilities (shared by CLI and scripts)."""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from pde_sim.core.logging import restore_stdout, setup_logging
from pde_sim.core.simulation import run_from_config


def collect_yaml_files(config_dir: Path, pattern: str = "**/*.yaml") -> list[Path]:
    """Collect YAML config files (recursive by default) and skip master configs."""
    yaml_files = list(config_dir.glob(pattern))
    if pattern.endswith(".yaml"):
        yaml_files.extend(config_dir.glob(pattern[:-5] + ".yml"))
    return sorted(
        p for p in set(yaml_files) if p.name not in {"master.yaml", "master.yml"}
    )


def _run_single(
    config_path: Path,
    output_dir: Path | None,
    seed: int | None,
    overwrite: bool,
    storage: str | None,
    keep_storage: bool | None,
    unique_suffix: bool | None,
    randomize: bool,
) -> dict:
    """Run a single simulation and return result info.

    Top-level function so it can be pickled for multiprocessing.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    preset = cfg.get("preset", "unknown")
    parameters = cfg.get("parameters", {})

    metadata = run_from_config(
        config_path=config_path,
        output_dir=output_dir,
        seed=seed,
        verbose=True,
        overwrite=overwrite,
        storage=storage,
        keep_storage=keep_storage,
        unique_suffix=unique_suffix,
        randomize=randomize,
    )

    return {
        "config_path": config_path,
        "preset": preset,
        "parameters": parameters,
        "folder_name": metadata.get("folder_name"),
        "diagnostics": metadata["diagnostics"],
    }


def _log_result(logger, index: int, total: int, result: dict) -> None:
    """Log the result of a single simulation."""
    logger.info("Simulation %d/%d: %s", index, total, result["config_path"].name)
    logger.info("  Preset: %s", result["preset"])
    logger.info("  Output: %s", result["folder_name"])

    diagnostics = result["diagnostics"]
    stagnant = diagnostics["stagnation"]["stagnant_fields"]
    nan_fields = diagnostics["nan_fields"]
    inf_fields = diagnostics["inf_fields"]
    _log_variability(logger, diagnostics["stagnation"])
    if stagnant:
        logger.warning("  Stagnant fields: %s", ", ".join(stagnant))
    if nan_fields:
        logger.warning("  Fields with NaN: %s", ", ".join(nan_fields))
    if inf_fields:
        logger.warning("  Fields with Inf: %s", ", ".join(inf_fields))


def _log_variability(logger, stagnation: dict) -> None:
    """Log per-field variability diagnostics."""
    fields = stagnation.get("fields", {})
    if not fields:
        return

    threshold_pct = float(stagnation.get("variability_threshold_percent", 0.0))
    logger.info("  Field variability (%% of range):")
    for field_name in sorted(fields):
        info = fields[field_name]
        variability_pct = float(
            info.get("variability_percent", info.get("max_relative_change", 0.0) * 100.0)
        )
        final_variability_pct = float(
            info.get(
                "final_variability_percent",
                info.get("final_relative_change", 0.0) * 100.0,
            )
        )
        logger.info(
            "    %s: %.6g%% (final: %.6g%%, threshold: %.6g%%)",
            field_name,
            variability_pct,
            final_variability_pct,
            threshold_pct,
        )
        if (not info.get("stagnant", False)) and info.get("variability_below_threshold", False):
            logger.warning(
                "  Field '%s' final variability %.6g%% is below threshold %.6g%%",
                field_name,
                final_variability_pct,
                threshold_pct,
            )


def run_batch(
    *,
    config_dir: Path,
    start_index: int = 1,
    log_file: Path | None = None,
    quiet: bool = False,
    pattern: str = "**/*.yaml",
    output_dir: Path | None = None,
    seed: int | None = None,
    storage: str | None = None,
    keep_storage: bool | None = None,
    unique_suffix: bool | None = False,
    overwrite: bool = False,
    continue_on_error: bool = True,
    randomize: bool = False,
    num_processes: int = 1,
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

    # Apply start_index filter
    indexed_configs = [
        (i, path)
        for i, path in enumerate(config_files, start=1)
        if i >= start_index
    ]

    total = len(config_files)
    logger.info("Found %d config files in %s", total, config_dir)
    logger.info("Running %d configs (starting from index %d)", len(indexed_configs), start_index)
    if num_processes > 1:
        logger.info("Using %d parallel processes", num_processes)
    if output_dir is not None:
        logger.info("Overriding output dir: %s", output_dir)
    if storage is not None:
        logger.info("Overriding output storage: %s", storage)
    if log_file is not None:
        logger.info("Logging to: %s", log_file)

    common_kwargs = dict(
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
        storage=storage,
        keep_storage=keep_storage,
        unique_suffix=unique_suffix,
        randomize=randomize,
    )

    if num_processes > 1:
        ok, failed = _run_parallel(
            logger, indexed_configs, total, common_kwargs,
            num_processes=num_processes,
            continue_on_error=continue_on_error,
        )
    else:
        ok, failed = _run_sequential(
            logger, indexed_configs, total, common_kwargs,
            continue_on_error=continue_on_error,
        )

    logger.info("=" * 60)
    logger.info("Batch complete: %d successful, %d failed", ok, failed)
    logger.info("=" * 60)

    restore_stdout()

    if failed and not continue_on_error:
        sys.stderr.write("Batch stopped due to failure\n")

    return ok, failed


def _run_sequential(
    logger,
    indexed_configs: list[tuple[int, Path]],
    total: int,
    common_kwargs: dict,
    *,
    continue_on_error: bool,
) -> tuple[int, int]:
    """Run simulations sequentially."""
    ok = 0
    failed = 0

    for i, config_path in indexed_configs:
        logger.info("=" * 60)
        logger.info("Simulation %d/%d: %s", i, total, config_path.name)

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            logger.info("Preset: %s", cfg.get("preset", "unknown"))
            logger.info("Parameters: %s", cfg.get("parameters", {}))

            metadata = run_from_config(
                config_path=config_path,
                verbose=True,
                **common_kwargs,
            )
            logger.info("Output: %s", metadata.get("folder_name"))

            diagnostics = metadata["diagnostics"]
            stagnant = diagnostics["stagnation"]["stagnant_fields"]
            nan_fields = diagnostics["nan_fields"]
            inf_fields = diagnostics["inf_fields"]
            _log_variability(logger, diagnostics["stagnation"])
            if stagnant:
                logger.warning("Stagnant fields: %s", ", ".join(stagnant))
            if nan_fields:
                logger.warning("Fields with NaN: %s", ", ".join(nan_fields))
            if inf_fields:
                logger.warning("Fields with Inf: %s", ", ".join(inf_fields))

            ok += 1
        except Exception as e:
            logger.error("Simulation %d (%s) failed: %s", i, config_path.name, e)
            failed += 1
            if not continue_on_error:
                break

    return ok, failed


def _run_parallel(
    logger,
    indexed_configs: list[tuple[int, Path]],
    total: int,
    common_kwargs: dict,
    *,
    num_processes: int,
    continue_on_error: bool,
) -> tuple[int, int]:
    """Run simulations in parallel using multiple processes."""
    ok = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all jobs
        future_to_index = {}
        for i, config_path in indexed_configs:
            future = executor.submit(_run_single, config_path, **common_kwargs)
            future_to_index[future] = (i, config_path)

        # Collect results as they complete
        for future in as_completed(future_to_index):
            i, config_path = future_to_index[future]
            try:
                result = future.result()
                _log_result(logger, i, total, result)
                ok += 1
            except Exception as e:
                logger.error("Simulation %d (%s) failed: %s", i, config_path.name, e)
                failed += 1
                if not continue_on_error:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    return ok, failed
