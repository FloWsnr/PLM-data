"""Batch execution utilities (shared by CLI and scripts)."""

import contextlib
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from pde_sim.core.logging import restore_stdout, setup_logging
from pde_sim.core.simulation import run_from_config


def _resolve_log_paths(log_file: Path | None) -> tuple[Path | None, Path | None]:
    """Resolve the main batch log file and per-simulation log directory."""
    if log_file is None:
        return None, None

    main_log_file = Path(log_file)
    simulation_log_dir = main_log_file.parent
    return main_log_file, simulation_log_dir


def _sanitize_log_name(value: str) -> str:
    """Sanitize a value so it can be used as part of a file name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "config"


def _get_simulation_log_file(
    simulation_log_dir: Path | None,
    index: int,
    config_path: Path,
    main_log_file: Path | None,
) -> Path | None:
    """Build a per-simulation log file path."""
    if simulation_log_dir is None:
        return None

    safe_name = _sanitize_log_name(config_path.stem)
    stem_prefix = f"{main_log_file.stem}_" if main_log_file is not None else ""
    return simulation_log_dir / f"{stem_prefix}{index:04d}_{safe_name}.log"


def _create_simulation_logger(log_handle) -> logging.Logger:
    """Create a logger that writes only to a simulation log handle."""
    logger = logging.getLogger("pde_sim.batch.simulation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(log_handle)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _close_logger_handlers(logger: logging.Logger) -> None:
    """Close and detach all handlers from a logger."""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        except OSError:
            pass
        logger.removeHandler(handler)


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
    simulation_log_file: Path | None,
    verbose: bool,
) -> dict:
    """Run a single simulation and return result info.

    Top-level function so it can be pickled for multiprocessing.
    """
    simulation_logger = None
    log_handle = None
    stdout_cm = contextlib.nullcontext()
    stderr_cm = contextlib.nullcontext()
    if simulation_log_file is not None:
        simulation_log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(simulation_log_file, "w", buffering=1)
        simulation_logger = _create_simulation_logger(log_handle)
        stdout_cm = contextlib.redirect_stdout(log_handle)
        stderr_cm = contextlib.redirect_stderr(log_handle)

    try:
        with stdout_cm, stderr_cm:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}

            preset = cfg.get("preset", "unknown")
            parameters = cfg.get("parameters", {})

            if simulation_logger is not None:
                simulation_logger.info("Config: %s", config_path)
                simulation_logger.info("Preset: %s", preset)
                simulation_logger.info("Parameters: %s", parameters)

            metadata = run_from_config(
                config_path=config_path,
                output_dir=output_dir,
                seed=seed,
                verbose=verbose,
                overwrite=overwrite,
                storage=storage,
                keep_storage=keep_storage,
                unique_suffix=unique_suffix,
                randomize=randomize,
            )

            diagnostics = metadata["diagnostics"]
            if simulation_logger is not None:
                simulation_logger.info("Output: %s", metadata.get("folder_name"))
                stagnant = diagnostics["stagnation"]["stagnant_fields"]
                nan_fields = diagnostics["nan_fields"]
                inf_fields = diagnostics["inf_fields"]
                _log_variability(simulation_logger, diagnostics["stagnation"])
                if stagnant:
                    simulation_logger.warning("Stagnant fields: %s", ", ".join(stagnant))
                if nan_fields:
                    simulation_logger.warning("Fields with NaN: %s", ", ".join(nan_fields))
                if inf_fields:
                    simulation_logger.warning("Fields with Inf: %s", ", ".join(inf_fields))

            return {
                "config_path": config_path,
                "preset": preset,
                "parameters": parameters,
                "folder_name": metadata.get("folder_name"),
                "diagnostics": diagnostics,
                "simulation_log_file": simulation_log_file,
            }
    finally:
        if simulation_logger is not None:
            _close_logger_handlers(simulation_logger)
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass


def _log_result(logger, index: int, total: int, result: dict) -> None:
    """Log the result of a single simulation."""
    logger.info("Simulation %d/%d: %s", index, total, result["config_path"].name)
    logger.info("  Preset: %s", result["preset"])
    logger.info("  Output: %s", result["folder_name"])
    if result.get("simulation_log_file") is not None:
        logger.info("  Log: %s", result["simulation_log_file"])

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
    main_log_file, simulation_log_dir = _resolve_log_paths(log_file)
    logger = setup_logging(
        log_file=main_log_file,
        console=not quiet,
        capture_stdout=False,
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
    if main_log_file is not None:
        logger.info("Main batch log: %s", main_log_file)
        logger.info("Per-simulation logs directory: %s", simulation_log_dir)

    common_kwargs = dict(
        output_dir=output_dir,
        seed=seed,
        overwrite=overwrite,
        storage=storage,
        keep_storage=keep_storage,
        unique_suffix=unique_suffix,
        randomize=randomize,
    )
    run_verbose = (not quiet) or (simulation_log_dir is not None)

    if num_processes > 1:
        ok, failed = _run_parallel(
            logger, indexed_configs, total, common_kwargs,
            num_processes=num_processes,
            continue_on_error=continue_on_error,
            simulation_log_dir=simulation_log_dir,
            run_verbose=run_verbose,
            main_log_file=main_log_file,
        )
    else:
        ok, failed = _run_sequential(
            logger, indexed_configs, total, common_kwargs,
            continue_on_error=continue_on_error,
            simulation_log_dir=simulation_log_dir,
            run_verbose=run_verbose,
            main_log_file=main_log_file,
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
    simulation_log_dir: Path | None,
    run_verbose: bool,
    main_log_file: Path | None,
) -> tuple[int, int]:
    """Run simulations sequentially."""
    ok = 0
    failed = 0

    for i, config_path in indexed_configs:
        logger.info("=" * 60)
        simulation_log_file = _get_simulation_log_file(
            simulation_log_dir, i, config_path, main_log_file
        )
        logger.info("Queueing simulation %d/%d: %s", i, total, config_path.name)
        if simulation_log_file is not None:
            logger.info("  Simulation log file: %s", simulation_log_file)

        try:
            result = _run_single(
                config_path=config_path,
                simulation_log_file=simulation_log_file,
                verbose=run_verbose,
                **common_kwargs,
            )
            _log_result(logger, i, total, result)
            ok += 1
        except Exception as e:
            logger.error("Simulation %d (%s) failed: %s", i, config_path.name, e)
            if simulation_log_file is not None:
                logger.error("  Simulation log file: %s", simulation_log_file)
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
    simulation_log_dir: Path | None,
    run_verbose: bool,
    main_log_file: Path | None,
) -> tuple[int, int]:
    """Run simulations in parallel using multiple processes."""
    ok = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all jobs
        future_to_index = {}
        for i, config_path in indexed_configs:
            simulation_log_file = _get_simulation_log_file(
                simulation_log_dir, i, config_path, main_log_file
            )
            logger.info("Queueing simulation %d/%d: %s", i, total, config_path.name)
            if simulation_log_file is not None:
                logger.info("  Simulation log file: %s", simulation_log_file)

            future = executor.submit(
                _run_single,
                config_path,
                simulation_log_file=simulation_log_file,
                verbose=run_verbose,
                **common_kwargs,
            )
            future_to_index[future] = (i, config_path, simulation_log_file)

        # Collect results as they complete
        for future in as_completed(future_to_index):
            i, config_path, simulation_log_file = future_to_index[future]
            try:
                result = future.result()
                _log_result(logger, i, total, result)
                ok += 1
            except Exception as e:
                logger.error("Simulation %d (%s) failed: %s", i, config_path.name, e)
                if simulation_log_file is not None:
                    logger.error("  Simulation log file: %s", simulation_log_file)
                failed += 1
                if not continue_on_error:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    return ok, failed
