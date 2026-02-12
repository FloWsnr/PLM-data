#!/usr/bin/env python3
"""Batch runner for PDE simulations from a directory of YAML config files.

This script is kept for compatibility with existing Slurm workflows, but the
implementation lives in `pde_sim.core.batch` so the CLI and scripts share code.
"""

import argparse
import sys
from pathlib import Path

from pde_sim.core.batch import run_batch


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
        default="**/*.yaml",
        help="Glob pattern for matching config files. Default: **/*.yaml",
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
