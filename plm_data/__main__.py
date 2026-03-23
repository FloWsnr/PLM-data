"""CLI entry point: python -m plm_data"""

import argparse
import sys


def cmd_run(args):
    from plm_data.core.runner import SimulationRunner

    runner = SimulationRunner.from_yaml(args.config)
    runner.run(verbose=True)


def cmd_list(args):
    from plm_data.presets import list_presets

    presets = list_presets()
    if not presets:
        print("No presets registered.")
        return

    by_category: dict[str, list] = {}
    for name, cls in sorted(presets.items()):
        meta = cls().metadata
        by_category.setdefault(meta.category, []).append(meta)

    for category, metas in sorted(by_category.items()):
        print(f"\n{category}:")
        for meta in metas:
            state = "steady" if meta.steady_state else "transient"
            dims = ", ".join(str(d) + "D" for d in meta.supported_dimensions)
            print(f"  {meta.name:20s}  [{state}, {dims}]  {meta.description}")


def cmd_show(args):
    import yaml

    from plm_data.presets import list_presets

    presets = list_presets()
    name = args.preset
    if name not in presets:
        print(f"Unknown preset '{name}'. Use 'list' to see available presets.")
        sys.exit(1)

    meta = presets[name]().metadata
    print(yaml.dump(meta.recommended_config, default_flow_style=False, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(prog="plm_data", description="PDE simulation data generation")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run a simulation from a YAML config")
    p_run.add_argument("config", help="Path to YAML config file")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="List available PDE presets")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show recommended config for a preset")
    p_show.add_argument("preset", help="Name of the preset")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
