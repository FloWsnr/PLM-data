#!/usr/bin/env python3
"""Batch runner for PDE simulations from a parameters CSV file."""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pde_sim.core.logging import restore_stdout, setup_logging
from pde_sim.core.simulation import run_from_config


# Mapping from CSV column names to config paths
SPECIAL_COLUMNS = {
    "BC_x": ("bc", "x"),
    "BC_y": ("bc", "y"),
    "init": ("init", "type"),
    "solver": ("solver",),
    "dt": ("dt",),
    "t_end": ("t_end",),
    "resolution": ("resolution",),
    "domain_size": ("domain_size",),
    "seed": ("seed",),
    "backend": ("backend",),
    "adaptive": ("adaptive",),
    "tolerance": ("tolerance",),
    "colormap": ("output", "colormap"),
    "field_to_plot": ("output", "field_to_plot"),
    "frames_per_save": ("output", "frames_per_save"),
    "save_array": ("output", "save_array"),
    "show_vectors": ("output", "show_vectors"),
    "vector_density": ("output", "vector_density"),
    "notes": None,  # Skip notes column
}


def load_base_config(config_path: Path) -> dict:
    """Load the base YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_value(value: str) -> Any:
    """Parse a string value to the appropriate type."""
    if value == "":
        return None

    # Try boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float (handles scientific notation like 2e-05)
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def set_nested_value(config: dict, path: tuple, value: any):
    """Set a value in a nested dict using a tuple path."""
    current = config
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def apply_csv_row(base_config: dict, headers: list, row: list) -> dict:
    """Apply CSV row values to a copy of the base config."""
    import copy

    config = copy.deepcopy(base_config)

    for header, value in zip(headers, row):
        value = value.strip()
        if not value:  # Skip empty values
            continue

        parsed_value = parse_value(value)
        if parsed_value is None:
            continue

        if header in SPECIAL_COLUMNS:
            path = SPECIAL_COLUMNS[header]
            if path is not None:  # Skip 'notes' column
                set_nested_value(config, path, parsed_value)
        else:
            # Assume it's a PDE parameter
            if "parameters" not in config:
                config["parameters"] = {}
            config["parameters"][header] = parsed_value

    return config


def run_batch(
    base_config_path: Path,
    params_csv_path: Path,
    start_row: int = 1,
    temp_dir: Path | None = None,
    log_file: Path | None = None,
    quiet: bool = False,
):
    """Run batch simulations from CSV parameters.

    Args:
        base_config_path: Path to base YAML config file.
        params_csv_path: Path to parameters CSV file.
        start_row: Start from row N (1-indexed, excluding header).
        temp_dir: Directory for temporary config files.
        log_file: Optional path to log file. If provided, all output is logged to this file.
        quiet: If True, suppress console output (only log to file if specified).
    """
    # Setup logging - capture stdout to get py-pde progress output in log file
    logger = setup_logging(
        log_file=log_file,
        console=not quiet,
        capture_stdout=log_file is not None,
    )

    base_config = load_base_config(base_config_path)

    if temp_dir is None:
        temp_dir = Path("/tmp/pde_sim_batch")
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    with open(params_csv_path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)  # First row is header
        rows = list(reader)

    total_rows = len(rows)
    logger.info(f"Found {total_rows} simulation configurations in CSV")
    logger.info(f"Starting from row {start_row}")
    if log_file:
        logger.info(f"Logging to: {log_file}")

    # Process rows starting from start_row (1-indexed)
    successful = 0
    failed = 0

    for i, row in enumerate(rows, start=1):
        if i < start_row:
            continue

        # Get notes for display if available
        notes_idx = headers.index("notes") if "notes" in headers else None
        notes = (
            row[notes_idx].strip()
            if notes_idx is not None and notes_idx < len(row)
            else ""
        )

        logger.info("=" * 60)
        logger.info(f"Simulation {i}/{total_rows}")
        if notes:
            logger.info(f"Notes: {notes}")

        try:
            # Generate modified config
            config = apply_csv_row(base_config, headers, row)

            # Generate run identifier (matches folder naming in SimulationRunner)
            preset_name = config.get("preset", "unknown")
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            run_id = f"{preset_name}/{run_timestamp}"
            logger.info(f"Run ID: {run_id}")

            # Write temporary config file
            temp_config_path = temp_dir / f"config_{i:05d}.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            # Log key parameters
            logger.info(f"Parameters: {config.get('parameters', {})}")
            logger.info(
                f"BC: x={config.get('bc', {}).get('x', 'periodic')}, "
                f"y={config.get('bc', {}).get('y', 'periodic')}"
            )
            logger.info(f"Init: {config.get('init', {}).get('type', 'unknown')}")
            logger.info(f"Time: t_end={config.get('t_end')}, dt={config.get('dt')}")

            # Run simulation with verbose=True to capture py-pde progress
            # Output goes to stdout which is captured to log file if specified
            metadata = run_from_config(
                config_path=temp_config_path,
                verbose=True,
            )

            logger.info(f"Output: {metadata['folder_name']}")
            successful += 1

        except Exception as e:
            logger.error(f"Simulation {i} failed: {e}")
            failed += 1
            continue

    logger.info("=" * 60)
    logger.info(f"Batch complete: {successful} successful, {failed} failed")
    logger.info("=" * 60)

    # Restore stdout/stderr if we were capturing
    restore_stdout()

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Run batch PDE simulations from a parameters CSV file"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Path to base YAML config file",
    )
    parser.add_argument(
        "--params-csv",
        type=Path,
        required=True,
        help="Path to parameters CSV file",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="Start from row N (1-indexed, excluding header). Default: 1",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Directory for temporary config files",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file. If provided, all output is logged to this file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (use with --log-file to only log to file)",
    )

    args = parser.parse_args()

    successful, failed = run_batch(
        base_config_path=args.base_config,
        params_csv_path=args.params_csv,
        start_row=args.start_row,
        temp_dir=args.temp_dir,
        log_file=args.log_file,
        quiet=args.quiet,
    )

    # Exit with error if any failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
