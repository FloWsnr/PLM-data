"""Simulation configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


def _require(raw: dict, key: str, context: str = "config"):
    """Require a key in a dict, raising a clear error if missing."""
    if key not in raw:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return raw[key]


@dataclass
class DomainConfig:
    """Domain/mesh configuration.

    `type` selects the geometry (e.g. "rectangle", "channel_with_cylinder").
    `params` holds all type-specific settings (size, mesh_resolution, etc.).
    """

    type: str
    params: dict


@dataclass
class ICConfig:
    """Initial condition configuration."""

    type: str
    params: dict


@dataclass
class OutputConfig:
    """Output configuration."""

    path: Path
    num_frames: int
    formats: list[str]


@dataclass
class SolverConfig:
    """PETSc solver options (keys without prefix)."""

    options: dict[str, str]


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    preset: str
    parameters: dict[str, float]
    domain: DomainConfig
    output_resolution: list[int]
    initial_condition: ICConfig | None
    output: OutputConfig
    solver: SolverConfig
    dt: float | None = None
    t_end: float | None = None
    seed: int | None = None


def load_config(path: str | Path) -> SimulationConfig:
    """Load a simulation config from a YAML file.

    All fields must be explicitly specified — no hidden defaults.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Required top-level fields
    preset = _require(raw, "preset")
    parameters = _require(raw, "parameters")
    output_resolution = _require(raw, "output_resolution")

    # Domain section (required)
    domain_raw = _require(raw, "domain")
    domain_type = _require(domain_raw, "type", "domain")
    domain_params = {k: v for k, v in domain_raw.items() if k != "type"}
    domain = DomainConfig(type=domain_type, params=domain_params)

    # Output section (required)
    output_raw = _require(raw, "output")
    output = OutputConfig(
        path=Path(_require(output_raw, "path", "output")),
        num_frames=_require(output_raw, "num_frames", "output"),
        formats=_require(output_raw, "formats", "output"),
    )

    # Solver options (required)
    solver_raw = _require(raw, "solver")
    if not isinstance(solver_raw, dict):
        raise ValueError("'solver' must be a mapping of PETSc option names to values")
    solver = SolverConfig(options={str(k): str(v) for k, v in solver_raw.items()})

    # Initial condition (optional — steady-state presets don't need it)
    ic_raw = raw.get("initial_condition")
    ic = None
    if ic_raw is not None:
        ic = ICConfig(
            type=_require(ic_raw, "type", "initial_condition"),
            params=_require(ic_raw, "params", "initial_condition"),
        )

    return SimulationConfig(
        preset=preset,
        parameters=parameters,
        domain=domain,
        output_resolution=output_resolution,
        initial_condition=ic,
        output=output,
        solver=solver,
        dt=raw.get("dt"),
        t_end=raw.get("t_end"),
        seed=raw.get("seed"),
    )
