"""CLI entry point: python -m plm_data."""

import argparse
import logging


def _seed(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def cmd_random(args: argparse.Namespace) -> dict[str, object]:
    """Run one fully sampled random simulation."""
    from mpi4py import MPI

    from plm_data.core.runner import run_random_simulation

    level = getattr(logging, args.log_level)
    summary = run_random_simulation(
        seed=args.seed,
        output_root=args.output_dir,
        console_level=level,
    )
    if MPI.COMM_WORLD.rank == 0:
        print(f"Output: {summary['output_dir']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plm_data",
        description="Generate one random 2D time-dependent PDE simulation.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=_seed,
        help="Required deterministic simulation seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output root. Defaults to ./output.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Console log level (file always logs DEBUG).",
    )
    args = parser.parse_args()
    cmd_random(args)


if __name__ == "__main__":
    main()
