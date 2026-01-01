"""CLI entry point for pde_sim."""

import argparse
import sys
from pathlib import Path

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

    args = parser.parse_args()

    if args.command == "run":
        run_simulation(args)
    elif args.command == "list":
        list_presets(args)
    elif args.command == "info":
        show_preset_info(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_simulation(args):
    """Run a simulation from config."""
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        metadata = run_from_config(
            config_path=args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print(f"\nOutput saved to: {metadata['folder_name']}")

    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)


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
        bounds = ""
        if param.min_value is not None or param.max_value is not None:
            lo = param.min_value if param.min_value is not None else "-inf"
            hi = param.max_value if param.max_value is not None else "inf"
            bounds = f" [{lo}, {hi}]"
        print(f"  {param.name}: {param.default}{bounds}")
        print(f"      {param.description}")
    print()

    if meta.reference:
        print(f"Reference: {meta.reference}")


if __name__ == "__main__":
    main()
