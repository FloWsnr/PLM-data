"""YAML configuration parsing and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Colormap cycle for auto-assigning to fields
COLORMAP_CYCLE = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "coolwarm",
    "twilight",
    "RdBu",
    "Spectral",
]


@dataclass
class OutputConfig:
    """Output configuration.

    All fields from the PDE are always output. Colormaps are auto-assigned
    from COLORMAP_CYCLE based on field order.
    """

    path: Path
    num_frames: int = 100  # Total number of frames to save
    format: str = "png"  # Output format: "png", "mp4", or "numpy"
    fps: int = 30  # Frame rate for MP4 output


@dataclass
class BoundaryConfig:
    """Boundary condition configuration.

    Uses py-pde notation for sides:
    - 1D: x- (left), x+ (right)
    - 2D: x- (left), x+ (right), y- (bottom), y+ (top)
    - 3D: x- (left), x+ (right), y- (bottom), y+ (top), z- (back), z+ (front)

    BC types:
    - `periodic` - periodic boundary
    - `neumann:VALUE` - fixed derivative (e.g., neumann:0)
    - `dirichlet:VALUE` - fixed value (e.g., dirichlet:0)

    Per-field overrides only need to specify sides that differ from default.
    """

    x_minus: str = "periodic"  # left
    x_plus: str = "periodic"  # right
    y_minus: str | None = None  # bottom (required for 2D+)
    y_plus: str | None = None  # top (required for 2D+)
    z_minus: str | None = None  # back (required for 3D)
    z_plus: str | None = None  # front (required for 3D)
    fields: dict[str, dict[str, str]] | None = None  # field -> {side: bc}

    def is_simple(self) -> bool:
        """Check if this is a simple (non-per-field) BC config."""
        return self.fields is None or len(self.fields) == 0

    def validate_for_ndim(self, ndim: int) -> None:
        """Validate boundaries are specified for given dimension.

        Args:
            ndim: Number of spatial dimensions (1, 2, or 3).

        Raises:
            ValueError: If required boundaries are not specified.
        """
        if ndim >= 2:
            if self.y_minus is None or self.y_plus is None:
                raise ValueError(
                    f"{ndim}D simulation requires y- and y+ boundary conditions"
                )
        if ndim >= 3:
            if self.z_minus is None or self.z_plus is None:
                raise ValueError(
                    "3D simulation requires z- and z+ boundary conditions"
                )

    def get_field_bc(self, field_name: str, ndim: int = 2) -> dict[str, str]:
        """Get BC dict for a field, merging with defaults.

        Args:
            field_name: Name of the field.
            ndim: Number of spatial dimensions.

        Returns:
            Dict with boundary keys containing BC strings.
        """
        default: dict[str, str] = {
            "x-": self.x_minus,
            "x+": self.x_plus,
        }
        if ndim >= 2 and self.y_minus is not None and self.y_plus is not None:
            default["y-"] = self.y_minus
            default["y+"] = self.y_plus
        if ndim >= 3 and self.z_minus is not None and self.z_plus is not None:
            default["z-"] = self.z_minus
            default["z+"] = self.z_plus

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
    resolution: list[int]  # [nx] for 1D, [nx, ny] for 2D, [nx, ny, nz] for 3D
    bc: BoundaryConfig
    output: OutputConfig
    seed: int | None = None
    domain_size: list[float] = field(default_factory=lambda: [1.0, 1.0])  # Matching dimensions
    backend: str = "auto"  # Options: "auto", "numpy", "numba"
    adaptive: bool = True  # Enable adaptive time-stepping
    tolerance: float = 1e-4  # Error tolerance for adaptive stepping

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.resolution)


def _parse_bc_config(bc_raw: dict, ndim: int = 2) -> BoundaryConfig:
    """Parse boundary condition configuration from raw dict.

    Expected format for 1D:
    ```yaml
    bc:
      x-: periodic
      x+: periodic
    ```

    Expected format for 2D:
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

    Expected format for 3D:
    ```yaml
    bc:
      x-: periodic
      x+: periodic
      y-: neumann:0
      y+: neumann:0
      z-: neumann:0
      z+: neumann:0
    ```
    """
    if not bc_raw:
        # Return defaults based on dimension
        if ndim == 1:
            return BoundaryConfig()
        elif ndim == 2:
            return BoundaryConfig(y_minus="periodic", y_plus="periodic")
        else:  # 3D
            return BoundaryConfig(
                y_minus="periodic", y_plus="periodic",
                z_minus="periodic", z_plus="periodic"
            )

    return BoundaryConfig(
        x_minus=bc_raw.get("x-", "periodic"),
        x_plus=bc_raw.get("x+", "periodic"),
        y_minus=bc_raw.get("y-"),  # None for 1D
        y_plus=bc_raw.get("y+"),  # None for 1D
        z_minus=bc_raw.get("z-"),  # None for 1D/2D
        z_plus=bc_raw.get("z+"),  # None for 1D/2D
        fields=bc_raw.get("fields"),
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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

    # Try to find master config relative to config file
    # e.g., configs/biology/bacteria_advection/default.yaml -> configs/master.yaml
    config_dir = path.parent
    master_path = None
    for _ in range(5):  # Walk up to 5 levels
        candidate = config_dir / "master.yaml"
        if candidate.exists():
            master_path = candidate
            break
        config_dir = config_dir.parent

    # Merge master config if found
    if master_path:
        with open(master_path) as f:
            master_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(master_raw, raw)

    # Parse nested configs
    init_config = InitialConditionConfig(
        type=raw["init"]["type"],
        params=raw["init"].get("params", {}),
    )

    # Parse resolution as list (supports 1D, 2D, 3D)
    resolution_raw = raw["resolution"]
    if not isinstance(resolution_raw, list):
        raise ValueError(
            f"resolution must be a list like [128, 128], got {type(resolution_raw).__name__}: {resolution_raw}"
        )
    resolution = [int(r) for r in resolution_raw]

    ndim = len(resolution)
    if ndim not in (1, 2, 3):
        raise ValueError(f"resolution must have 1, 2, or 3 elements, got {ndim}")

    # Parse domain_size as list (must match resolution dimensions)
    domain_size_raw = raw.get("domain_size")
    if domain_size_raw is None:
        # Default to unit size in each dimension
        domain_size = [1.0] * ndim
    elif not isinstance(domain_size_raw, list):
        raise ValueError(
            f"domain_size must be a list like [10.0, 10.0], got {type(domain_size_raw).__name__}: {domain_size_raw}"
        )
    else:
        domain_size = [float(d) for d in domain_size_raw]
        if len(domain_size) != ndim:
            raise ValueError(
                f"domain_size has {len(domain_size)} elements but resolution has {ndim}"
            )

    # Parse boundary config with dimension awareness
    bc_config = _parse_bc_config(raw.get("bc", {}), ndim)

    output_raw = raw.get("output", {})

    output_config = OutputConfig(
        path=Path(output_raw.get("path", "./output")),
        num_frames=output_raw.get("num_frames", 100),
        format=output_raw.get("format", "png"),
        fps=output_raw.get("fps", 30),
    )

    return SimulationConfig(
        preset=raw["preset"],
        parameters=raw.get("parameters", {}),
        init=init_config,
        solver=raw.get("solver", "euler"),
        t_end=raw["t_end"],
        dt=raw["dt"],
        resolution=resolution,
        bc=bc_config,
        output=output_config,
        seed=raw.get("seed"),
        domain_size=domain_size,
        backend=raw.get("backend", "numba"),
        adaptive=raw.get("adaptive", True),
        tolerance=raw.get("tolerance", 1e-4),
    )


def _bc_config_to_dict(bc: BoundaryConfig, ndim: int = 2) -> dict[str, Any]:
    """Convert BoundaryConfig to dictionary for serialization.

    Args:
        bc: BoundaryConfig object.
        ndim: Number of spatial dimensions.

    Returns:
        Dictionary representation of boundary conditions.
    """
    bc_dict: dict[str, Any] = {
        "x-": bc.x_minus,
        "x+": bc.x_plus,
    }

    if ndim >= 2 and bc.y_minus is not None:
        bc_dict["y-"] = bc.y_minus
        bc_dict["y+"] = bc.y_plus

    if ndim >= 3 and bc.z_minus is not None:
        bc_dict["z-"] = bc.z_minus
        bc_dict["z+"] = bc.z_plus

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
        "bc": _bc_config_to_dict(config.bc, config.ndim),
        "output": {
            "path": str(config.output.path),
            "num_frames": config.output.num_frames,
            "format": config.output.format,
            "fps": config.output.fps,
        },
        "seed": config.seed,
        "domain_size": config.domain_size,
        "backend": config.backend,
        "adaptive": config.adaptive,
        "tolerance": config.tolerance,
    }
