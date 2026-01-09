#!/usr/bin/env python3
"""Batch runner for PDE simulations from a directory of YAML config files."""

import argparse
import sys
from pathlib import Path

import yaml

from pde_sim.core.logging import restore_stdout, setup_logging
from pde_sim.core.simulation import run_from_config


def collect_yaml_files(config_dir: Path, pattern: str = "*.yaml") -> list[Path]:
    """Collect all YAML config files from a directory.

    Args:
        config_dir: Directory containing YAML config files.
        pattern: Glob pattern for matching files. Default: "*.yaml"

    Returns:
        Sorted list of YAML file paths.
    """
    yaml_files = list(config_dir.glob(pattern))
    # Also check for .yml extension
    yaml_files.extend(config_dir.glob(pattern.replace(".yaml", ".yml")))
    return sorted(yaml_files)


def run_batch(
    config_dir: Path,
    start_index: int = 1,
    log_file: Path | None = None,
    quiet: bool = False,
    pattern: str = "*.yaml",
):
    """Run batch simulations from a directory of YAML configs.

    Args:
        config_dir: Directory containing YAML config files.
        start_index: Start from config N (1-indexed).
        log_file: Optional path to log file. If provided, all output is logged to this file.
        quiet: If True, suppress console output (only log to file if specified).
        pattern: Glob pattern for matching config files. Default: "*.yaml"
    """
    # Setup logging - capture stdout to get py-pde progress output in log file
    logger = setup_logging(
        log_file=log_file,
        console=not quiet,
        capture_stdout=log_file is not None,
    )

    # Collect YAML files
    config_files = collect_yaml_files(config_dir, pattern)

    if not config_files:
        logger.error(f"No YAML config files found in {config_dir}")
        return 0, 0

    total_configs = len(config_files)
    logger.info(f"Found {total_configs} config files in {config_dir}")
    logger.info(f"Starting from index {start_index}")
    if log_file:
        logger.info(f"Logging to: {log_file}")

    # Process configs starting from start_index (1-indexed)
    successful = 0
    failed = 0

    for i, config_path in enumerate(config_files, start=1):
        if i < start_index:
            continue

        logger.info("=" * 60)
        logger.info(f"Simulation {i}/{total_configs}: {config_path.name}")

        try:
            # Load config to log key info
            with open(config_path) as f:
                config = yaml.safe_load(f)

            preset_name = config.get("preset", "unknown")
            logger.info(f"Preset: {preset_name}")
            logger.info(f"Parameters: {config.get('parameters', {})}")

            # Run simulation
            metadata = run_from_config(
                config_path=config_path,
                verbose=True,
            )

            logger.info(f"Output: {metadata['folder_name']}")
            successful += 1

        except Exception as e:
            logger.error(f"Simulation {i} ({config_path.name}) failed: {e}")
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
        description="Run batch PDE simulations from a directory of YAML config files"
    )
    parser.add_argument(
        "config_dir",
        type=Path,
        help="Directory containing YAML config files",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start from config N (1-indexed). Default: 1",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="Glob pattern for matching config files. Default: *.yaml",
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

    if not args.config_dir.is_dir():
        print(f"Error: {args.config_dir} is not a directory")
        sys.exit(1)

    successful, failed = run_batch(
        config_dir=args.config_dir,
        start_index=args.start_index,
        log_file=args.log_file,
        quiet=args.quiet,
        pattern=args.pattern,
    )

    # Exit with error if any failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
