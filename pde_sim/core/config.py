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
    """Boundary condition configuration.

    Uses py-pde notation for sides:
    - x- (left), x+ (right), y- (bottom), y+ (top)

    BC types:
    - `periodic` - periodic boundary
    - `neumann:VALUE` - fixed derivative (e.g., neumann:0)
    - `dirichlet:VALUE` - fixed value (e.g., dirichlet:0)

    Per-field overrides only need to specify sides that differ from default.
    """

    x_minus: str = "periodic"  # left
    x_plus: str = "periodic"  # right
    y_minus: str = "periodic"  # bottom
    y_plus: str = "periodic"  # top
    fields: dict[str, dict[str, str]] | None = None  # field -> {side: bc}

    def is_simple(self) -> bool:
        """Check if this is a simple (non-per-field) BC config."""
        return self.fields is None or len(self.fields) == 0

    def get_field_bc(self, field_name: str) -> dict[str, str]:
        """Get BC dict for a field, merging with defaults.

        Args:
            field_name: Name of the field.

        Returns:
            Dict with x-, x+, y-, y+ keys containing BC strings.
        """
        default = {
            "x-": self.x_minus,
            "x+": self.x_plus,
            "y-": self.y_minus,
            "y+": self.y_plus,
        }
        if self.fields and field_name in self.fields:
            default.update(self.fields[field_name])
        return default


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

    Expected format:
    ```yaml
    bc:
      x-: periodic
      x+: periodic
      y-: neumann:0
      y+: neumann:0
      fields:
        omega:
          y-: dirichlet:0
          y+: dirichlet:0
    ```
    """
    if not bc_raw:
        return BoundaryConfig()

    return BoundaryConfig(
        x_minus=bc_raw.get("x-", "periodic"),
        x_plus=bc_raw.get("x+", "periodic"),
        y_minus=bc_raw.get("y-", "periodic"),
        y_plus=bc_raw.get("y+", "periodic"),
        fields=bc_raw.get("fields"),
    )


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
    bc_dict: dict[str, Any] = {
        "x-": bc.x_minus,
        "x+": bc.x_plus,
        "y-": bc.y_minus,
        "y+": bc.y_plus,
    }

    if bc.fields:
        bc_dict["fields"] = bc.fields

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
