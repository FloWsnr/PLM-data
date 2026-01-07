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
class FieldBoundaryConfig:
    """Boundary conditions for a single field.

    Can specify either:
    - x/y (same BC for both sides of that axis), or
    - left/right/top/bottom (individual sides)

    Side-specific values override axis-level values.
    """

    x: str | None = None
    y: str | None = None
    left: str | None = None  # Overrides x for left side (x-)
    right: str | None = None  # Overrides x for right side (x+)
    top: str | None = None  # Overrides y for top (y+)
    bottom: str | None = None  # Overrides y for bottom (y-)


@dataclass
class BoundaryConfig:
    """Boundary condition configuration.

    Supports three modes:
    1. Simple: x/y strings (backward compatible)
    2. Per-field: fields dict mapping field names to FieldBoundaryConfig
    3. Mixed: default x/y plus per-field overrides
    """

    x: str = "periodic"
    y: str = "periodic"
    fields: dict[str, FieldBoundaryConfig] | None = None

    def is_simple(self) -> bool:
        """Check if this is a simple (non-per-field) BC config."""
        return self.fields is None or len(self.fields) == 0

    def get_field_bc(self, field_name: str) -> FieldBoundaryConfig:
        """Get BC for a specific field, falling back to defaults."""
        if self.fields and field_name in self.fields:
            field_bc = self.fields[field_name]
            # Fill in defaults for unspecified values
            return FieldBoundaryConfig(
                x=field_bc.x if field_bc.x is not None else self.x,
                y=field_bc.y if field_bc.y is not None else self.y,
                left=field_bc.left,
                right=field_bc.right,
                top=field_bc.top,
                bottom=field_bc.bottom,
            )
        return FieldBoundaryConfig(x=self.x, y=self.y)


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


def _parse_bc_config(bc_raw: dict) -> BoundaryConfig:
    """Parse boundary condition configuration from raw dict.

    Supports:
    - Simple format: {x: "periodic", y: "neumann"}
    - Per-field format: {x: "periodic", y: "periodic", fields: {u: {x: "dirichlet"}, ...}}
    """
    if not bc_raw:
        return BoundaryConfig()

    # Get default x/y BCs
    default_x = bc_raw.get("x", "periodic")
    default_y = bc_raw.get("y", "periodic")

    # Parse per-field BCs if present
    fields = None
    if "fields" in bc_raw and bc_raw["fields"]:
        fields = {}
        for field_name, field_bc in bc_raw["fields"].items():
            if isinstance(field_bc, dict):
                fields[field_name] = FieldBoundaryConfig(
                    x=field_bc.get("x"),
                    y=field_bc.get("y"),
                    left=field_bc.get("left"),
                    right=field_bc.get("right"),
                    top=field_bc.get("top"),
                    bottom=field_bc.get("bottom"),
                )

    return BoundaryConfig(x=default_x, y=default_y, fields=fields)


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

    bc_config = _parse_bc_config(raw.get("bc", {}))

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


def _bc_config_to_dict(bc: BoundaryConfig) -> dict[str, Any]:
    """Convert BoundaryConfig to dictionary for serialization."""
    bc_dict: dict[str, Any] = {"x": bc.x, "y": bc.y}

    if bc.fields:
        bc_dict["fields"] = {}
        for field_name, field_bc in bc.fields.items():
            field_dict: dict[str, str] = {}
            if field_bc.x is not None:
                field_dict["x"] = field_bc.x
            if field_bc.y is not None:
                field_dict["y"] = field_bc.y
            if field_bc.left is not None:
                field_dict["left"] = field_bc.left
            if field_bc.right is not None:
                field_dict["right"] = field_bc.right
            if field_bc.top is not None:
                field_dict["top"] = field_bc.top
            if field_bc.bottom is not None:
                field_dict["bottom"] = field_bc.bottom
            if field_dict:
                bc_dict["fields"][field_name] = field_dict

    return bc_dict


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
        "bc": _bc_config_to_dict(config.bc),
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
