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
class BCConfig:
    """Configuration for a single boundary condition.

    `type` is "dirichlet", "neumann", or "robin".
    `value` is a float, "param:name" string, or {type, params} dict for
    spatial fields (same system as source terms).

    For robin BCs (∂u/∂n + α*u = g):
      - `alpha` is the coefficient on u (float or "param:name")
      - `value` is g (the RHS, same format as other BC values)
    """

    type: str
    value: float | str | dict
    alpha: float | str | None = None


@dataclass
class SourceTermConfig:
    """Source term configuration.

    Uses the shared spatial field type system (constant, sine_product,
    gaussian_bump, none, custom).
    """

    type: str
    params: dict


@dataclass
class DomainConfig:
    """Domain/mesh configuration.

    `type` selects the geometry (e.g. "rectangle", "channel_with_cylinder").
    `params` holds all type-specific settings (size, mesh_resolution, etc.).
    Domain is pure geometry — boundary conditions are per-field at the
    SimulationConfig level.
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
    """Complete simulation configuration.

    boundary_conditions, source_terms, and initial_conditions are all
    per-field: keyed by field name (e.g. "u", "velocity", "pressure").
    """

    preset: str
    parameters: dict[str, float]
    domain: DomainConfig
    output_resolution: list[int]
    boundary_conditions: dict[str, dict[str, BCConfig]]
    source_terms: dict[str, SourceTermConfig]
    initial_conditions: dict[str, ICConfig]
    output: OutputConfig
    solver: SolverConfig
    dt: float | None = None
    t_end: float | None = None
    seed: int | None = None


def _parse_bc_spec(bc_spec: dict, context: str) -> BCConfig:
    """Parse a single BC spec dict into a BCConfig."""
    if not isinstance(bc_spec, dict):
        raise ValueError(
            f"Boundary condition in {context} must be a mapping "
            f"with 'type' and 'value' keys. Got: {bc_spec}"
        )
    bc_type = _require(bc_spec, "type", context)
    bc_value = _require(bc_spec, "value", context)
    bc_alpha = bc_spec.get("alpha")
    if bc_type == "robin" and bc_alpha is None:
        raise ValueError(
            f"Robin BC in {context} requires 'alpha' field. Got: {bc_spec}"
        )
    return BCConfig(type=bc_type, value=bc_value, alpha=bc_alpha)


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

    # Domain section (required, pure geometry)
    domain_raw = _require(raw, "domain")
    domain_type = _require(domain_raw, "type", "domain")
    domain_params = {k: v for k, v in domain_raw.items() if k != "type"}
    domain = DomainConfig(type=domain_type, params=domain_params)

    # Boundary conditions (required, per-field)
    bc_raw = _require(raw, "boundary_conditions")
    boundary_conditions = {}
    for field_name, field_bcs in bc_raw.items():
        bcs = {}
        for bc_name, bc_spec in field_bcs.items():
            ctx = f"boundary_conditions.{field_name}.{bc_name}"
            bcs[bc_name] = _parse_bc_spec(bc_spec, ctx)
        boundary_conditions[field_name] = bcs

    # Source terms (required, per-field)
    st_raw = _require(raw, "source_terms")
    source_terms = {}
    for field_name, st_spec in st_raw.items():
        source_terms[field_name] = SourceTermConfig(
            type=_require(st_spec, "type", f"source_terms.{field_name}"),
            params=st_spec.get("params", {}),
        )

    # Initial conditions (optional, per-field — steady-state presets use {})
    ic_raw = raw.get("initial_conditions", {})
    initial_conditions = {}
    for field_name, ic_spec in ic_raw.items():
        initial_conditions[field_name] = ICConfig(
            type=_require(ic_spec, "type", f"initial_conditions.{field_name}"),
            params=_require(ic_spec, "params", f"initial_conditions.{field_name}"),
        )

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

    return SimulationConfig(
        preset=preset,
        parameters=parameters,
        domain=domain,
        output_resolution=output_resolution,
        boundary_conditions=boundary_conditions,
        source_terms=source_terms,
        initial_conditions=initial_conditions,
        output=output,
        solver=solver,
        dt=raw.get("dt"),
        t_end=raw.get("t_end"),
        seed=raw.get("seed"),
    )
