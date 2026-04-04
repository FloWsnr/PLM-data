"""CLI entry point: python -m plm_data"""

import argparse
import logging
import sys


def cmd_run(args):
    from plm_data.core.runner import SimulationRunner

    level = getattr(logging, args.log_level)
    runner = SimulationRunner.from_yaml(args.config, args.output_dir, seed=args.seed)
    runner.run(console_level=level)


def cmd_list(_args):
    from plm_data.presets import list_presets

    presets = list_presets()
    if not presets:
        print("No presets registered.")
        return

    by_category: dict[str, list] = {}
    for _, cls in sorted(presets.items()):
        spec = cls().spec
        by_category.setdefault(spec.category, []).append(spec)

    for category, specs in sorted(by_category.items()):
        print(f"\n{category}:")
        for spec in specs:
            state = "steady" if spec.steady_state else "transient"
            dims = ", ".join(str(d) + "D" for d in spec.supported_dimensions)
            print(f"  {spec.name:20s}  [{state}, {dims}]  {spec.description}")


def main():
    parser = argparse.ArgumentParser(
        prog="plm_data", description="PDE simulation data generation"
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run a simulation from a YAML config")
    p_run.add_argument("config", help="Path to YAML config file")
    p_run.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory. Results are written to <dir>/<category>/<preset>",
    )
    p_run.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Console log level (file always logs DEBUG)",
    )
    p_run.add_argument(
        "--seed",
        type=int,
        help="Override the config seed for this run.",
    )
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="List available PDE presets")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
