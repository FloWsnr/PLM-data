"""YAML configuration parsing and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OutputConfig:
    """Output configuration."""

    path: Path
    frames_per_save: int = 10
    colormap: str = "turbo"
    field_to_plot: str | None = None


@dataclass
class BoundaryConfig:
    """Boundary condition configuration."""

    x: str = "periodic"
    y: str = "periodic"


@dataclass
class InitialConditionConfig:
    """Initial condition configuration."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    preset: str
    parameters: dict[str, float]
    init: InitialConditionConfig
    solver: str
    timesteps: int
    dt: float
    resolution: int
    bc: BoundaryConfig
    output: OutputConfig
    seed: int | None = None
    domain_size: float = 1.0  # Physical size of the domain


def load_config(path: Path | str) -> SimulationConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed SimulationConfig object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        KeyError: If required fields are missing.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Parse nested configs
    init_config = InitialConditionConfig(
        type=raw["init"]["type"],
        params=raw["init"].get("params", {}),
    )

    bc_config = BoundaryConfig(
        x=raw.get("bc", {}).get("x", "periodic"),
        y=raw.get("bc", {}).get("y", "periodic"),
    )

    output_raw = raw.get("output", {})
    output_config = OutputConfig(
        path=Path(output_raw.get("path", "./output")),
        frames_per_save=output_raw.get("frames_per_save", 10),
        colormap=output_raw.get("colormap", "turbo"),
        field_to_plot=output_raw.get("field_to_plot"),
    )

    return SimulationConfig(
        preset=raw["preset"],
        parameters=raw.get("parameters", {}),
        init=init_config,
        solver=raw.get("solver", "euler"),
        timesteps=raw["timesteps"],
        dt=raw["dt"],
        resolution=raw["resolution"],
        bc=bc_config,
        output=output_config,
        seed=raw.get("seed"),
        domain_size=raw.get("domain_size", 1.0),
    )


def config_to_dict(config: SimulationConfig) -> dict[str, Any]:
    """Convert a SimulationConfig back to a dictionary for serialization."""
    return {
        "preset": config.preset,
        "parameters": config.parameters,
        "init": {
            "type": config.init.type,
            "params": config.init.params,
        },
        "solver": config.solver,
        "timesteps": config.timesteps,
        "dt": config.dt,
        "resolution": config.resolution,
        "bc": {
            "x": config.bc.x,
            "y": config.bc.y,
        },
        "output": {
            "path": str(config.output.path),
            "frames_per_save": config.output.frames_per_save,
            "colormap": config.output.colormap,
            "field_to_plot": config.output.field_to_plot,
        },
        "seed": config.seed,
        "domain_size": config.domain_size,
    }
