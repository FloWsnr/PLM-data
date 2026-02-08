"""CLI entry point for pde_sim."""

import argparse
import sys
from pathlib import Path

from pde_sim.core.logging import restore_stdout, setup_logging
from pde_sim.core.overview import generate_overview
from pde_sim.core.batch import run_batch
from pde_sim.core.simulation import run_from_config
from pde_sim.pdes import get_pde_preset, get_presets_by_category


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="2D PDE Simulation Dataset Generator",
        prog="python -m pde_sim",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a simulation from config")
    run_parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the last numbered output folder instead of creating a new one",
    )
    run_parser.add_argument(
        "--unique-suffix",
        action="store_true",
        help="Append a short random suffix to the run folder name (helps avoid collisions)",
    )
    run_parser.add_argument(
        "--storage",
        type=str,
        choices=["memory", "file"],
        help="Output storage mode for py-pde frames (memory or file)",
    )
    run_parser.add_argument(
        "--keep-storage",
        action="store_true",
        help="Keep intermediate py-pde storage file when using --storage=file",
    )
    run_parser.add_argument(
        "--randomize-positions",
        action="store_true",
        help="Replace explicit IC position values with random, so the same config produces diverse spatial layouts",
    )
    run_parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file. If provided, all output is logged to this file.",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available presets")
    list_parser.add_argument(
        "--category",
        "-c",
        type=str,
        help="Filter by category",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show info about a preset")
    info_parser.add_argument(
        "preset",
        type=str,
        help="Name of the preset",
    )

    # Overview command
    overview_parser = subparsers.add_parser(
        "overview", help="Generate HTML overview of GIF simulations"
    )
    overview_parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing simulation outputs to scan",
    )
    overview_parser.add_argument(
        "--html",
        type=Path,
        help="Path for output HTML file (default: output_dir/overview.html)",
    )
    overview_parser.add_argument(
        "--title",
        type=str,
        default="Simulation Overview",
        help="Title for the HTML document",
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Run all configs in a directory (optionally with logging)"
    )
    batch_parser.add_argument(
        "config_dir",
        type=Path,
        help="Directory containing YAML config files",
    )
    batch_parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start from config N (1-indexed). Default: 1",
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="Glob pattern for matching config files. Default: *.yaml",
    )
    batch_parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file. If provided, all output is logged to this file.",
    )
    batch_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (use with --log-file to only log to file)",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory for all runs",
    )
    batch_parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed for all runs",
    )
    batch_parser.add_argument(
        "--storage",
        type=str,
        choices=["memory", "file"],
        help="Override output storage mode (memory or file)",
    )
    batch_parser.add_argument(
        "--keep-storage",
        action="store_true",
        help="Keep intermediate py-pde storage file when using --storage=file",
    )
    batch_parser.add_argument(
        "--no-unique-suffix",
        action="store_true",
        help="Disable unique run-name suffixes (default: enabled for batch)",
    )
    batch_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the last numbered output folder instead of creating a new one",
    )
    batch_parser.add_argument(
        "--randomize-positions",
        action="store_true",
        help="Replace explicit IC position values with random, so configs produce diverse spatial layouts",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_simulation(args)
    elif args.command == "list":
        list_presets(args)
    elif args.command == "info":
        show_preset_info(args)
    elif args.command == "overview":
        create_overview(args)
    elif args.command == "batch":
        run_batch_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_simulation(args):
    """Run a simulation from config."""
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Setup logging if log file is specified
    log_file = getattr(args, "log_file", None)
    if log_file is not None:
        setup_logging(
            log_file=log_file,
            console=not args.quiet,
            capture_stdout=True,
        )

    try:
        metadata = run_from_config(
            config_path=args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            verbose=not args.quiet,
            overwrite=args.overwrite,
            storage=args.storage,
            keep_storage=True if args.keep_storage else None,
            unique_suffix=True if args.unique_suffix else None,
            randomize_positions=args.randomize_positions,
        )

        if not args.quiet:
            print(f"\nOutput saved to: {metadata['folder_name']}")

    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)
    finally:
        # Restore stdout/stderr if we were capturing
        if log_file is not None:
            restore_stdout()


def list_presets(args):
    """List all available PDE presets."""
    print("Available PDE Presets")
    print("=" * 60)

    categories = get_presets_by_category()

    for category, presets in sorted(categories.items()):
        if args.category and args.category.lower() != category.lower():
            continue

        print(f"\n{category.upper().replace('-', ' ')}:")
        for name in presets:
            preset = get_pde_preset(name)
            desc = preset.metadata.description
            print(f"  {name:30s} - {desc}")

    print()


def show_preset_info(args):
    """Show detailed info about a preset."""
    try:
        preset = get_pde_preset(args.preset)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    meta = preset.metadata

    print(f"Preset: {meta.name}")
    print(f"Category: {meta.category}")
    print(f"Description: {meta.description}")
    print()

    print("Fields:")
    for name in meta.field_names:
        print(f"  - {name}")
    print()

    print("Equations:")
    for var, eq in meta.equations.items():
        print(f"  d{var}/dt = {eq}")
    print()

    print("Parameters:")
    for param in meta.parameters:
        print(f"  {param.name}: {param.description}")
    print()

    if meta.reference:
        print(f"Reference: {meta.reference}")


def create_overview(args):
    """Generate HTML overview of GIF simulations."""
    if not args.output_dir.exists():
        print(f"Error: Directory not found: {args.output_dir}")
        sys.exit(1)

    # Default HTML path is inside a self-contained overview/ subdirectory
    html_path = args.html if args.html else args.output_dir / "00_overview" / "overview.html"

    count = generate_overview(
        output_dir=args.output_dir,
        html_path=html_path,
        title=args.title,
    )

    if count == 0:
        print("No simulations with GIF files found.")
        sys.exit(1)

    print(f"Generated overview with {count} simulations: {html_path}")


def run_batch_command(args) -> None:
    """Run all YAML configs in a directory."""
    if not args.config_dir.is_dir():
        print(f"Error: {args.config_dir} is not a directory")
        sys.exit(1)

    ok, failed = run_batch(
        config_dir=args.config_dir,
        start_index=args.start_index,
        log_file=args.log_file,
        quiet=args.quiet,
        pattern=args.pattern,
        output_dir=args.output_dir,
        seed=args.seed,
        storage=args.storage,
        keep_storage=True if args.keep_storage else None,
        unique_suffix=(not args.no_unique_suffix),
        overwrite=args.overwrite,
        randomize_positions=args.randomize_positions,
    )

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
