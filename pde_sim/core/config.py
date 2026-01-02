"""YAML configuration parsing and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OutputConfig:
    """Output configuration."""

    path: Path
    num_frames: int = 100  # Total number of frames to save
    colormap: str = "turbo"
    field_to_plot: str | None = None
    save_array: bool = False  # Save trajectory as numpy array (.npy)
    show_vectors: bool = False  # Overlay vector arrows when using mag()
    vector_density: int = 16  # Number of arrows per axis (e.g., 16 = 16x16 grid)


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
    t_end: float
    dt: float
    resolution: int
    bc: BoundaryConfig
    output: OutputConfig
    seed: int | None = None
    domain_size: float = 1.0  # Physical size of the domain
    backend: str = "auto"  # Options: "auto", "numpy", "numba"
    adaptive: bool = True  # Enable adaptive time-stepping
    tolerance: float = 1e-4  # Error tolerance for adaptive stepping


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
        num_frames=output_raw.get("num_frames", 100),
        colormap=output_raw.get("colormap", "turbo"),
        field_to_plot=output_raw.get("field_to_plot"),
        save_array=output_raw.get("save_array", False),
        show_vectors=output_raw.get("show_vectors", False),
        vector_density=output_raw.get("vector_density", 16),
    )

    return SimulationConfig(
        preset=raw["preset"],
        parameters=raw.get("parameters", {}),
        init=init_config,
        solver=raw.get("solver", "euler"),
        t_end=raw["t_end"],
        dt=raw["dt"],
        resolution=raw["resolution"],
        bc=bc_config,
        output=output_config,
        seed=raw.get("seed"),
        domain_size=raw.get("domain_size", 1.0),
        backend=raw.get("backend", "numba"),
        adaptive=raw.get("adaptive", True),
        tolerance=raw.get("tolerance", 1e-4),
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
        "t_end": config.t_end,
        "dt": config.dt,
        "resolution": config.resolution,
        "bc": {
            "x": config.bc.x,
            "y": config.bc.y,
        },
        "output": {
            "path": str(config.output.path),
            "num_frames": config.output.num_frames,
            "colormap": config.output.colormap,
            "field_to_plot": config.output.field_to_plot,
            "save_array": config.output.save_array,
            "show_vectors": config.output.show_vectors,
            "vector_density": config.output.vector_density,
        },
        "seed": config.seed,
        "domain_size": config.domain_size,
        "backend": config.backend,
        "adaptive": config.adaptive,
        "tolerance": config.tolerance,
    }
