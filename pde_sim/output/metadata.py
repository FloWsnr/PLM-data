"""Metadata generation for simulation runs."""

from datetime import datetime, timezone
from typing import Any

from pde_sim.descriptions import get_description


def _get_generator_version() -> str:
    try:
        from importlib.metadata import version

        return version("pde-sim")
    except Exception:
        try:
            from pde_sim import __version__

            return __version__
        except Exception:
            return "unknown"


def _bc_to_metadata(bc: Any, ndim: int = 2) -> dict[str, Any]:
    result: dict[str, Any] = {"x-": bc.x_minus, "x+": bc.x_plus}

    if ndim >= 2 and bc.y_minus is not None:
        result["y-"] = bc.y_minus
        result["y+"] = bc.y_plus

    if ndim >= 3 and bc.z_minus is not None:
        result["z-"] = bc.z_minus
        result["z+"] = bc.z_plus

    if bc.fields:
        result["fields"] = bc.fields

    return result


def _create_visualization_metadata(config: Any, preset_metadata: Any) -> dict[str, Any]:
    from pde_sim.core.config import COLORMAP_CYCLE  # avoid circular import

    field_names = preset_metadata.field_names
    return {
        "whatToPlot": field_names,
        "colormaps": {
            name: COLORMAP_CYCLE[i % len(COLORMAP_CYCLE)]
            for i, name in enumerate(field_names)
        },
    }


def create_metadata(
    sim_id: str,
    preset_name: str,
    preset_metadata: Any,
    config: Any,
    total_time: float,
    frame_annotations: list[dict[str, Any]],
    solver_diagnostics: dict[str, Any] | None = None,
    wall_clock_duration: float | None = None,
) -> dict[str, Any]:
    solver_diagnostics = solver_diagnostics or {}

    dt_stats = solver_diagnostics.get("dt_statistics")
    if dt_stats:
        dt_stats = {k: float(v) if hasattr(v, "item") else v for k, v in dt_stats.items()}

    description = get_description(preset_name)

    spatial_steps = [
        config.domain_size[i] / config.resolution[i] for i in range(config.ndim)
    ]

    return {
        "id": sim_id,
        "preset": preset_name,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generatorVersion": _get_generator_version(),
        "equations": preset_metadata.equations,
        "boundaryConditions": _bc_to_metadata(config.bc, config.ndim),
        "initialConditions": config.init.type,
        "parameters": {
            "kinetic": config.parameters,
            "dt": config.dt,
            "spatialStep": spatial_steps,
            "domainSize": config.domain_size,
            "timesteppingScheme": config.solver.title(),
            "numSpecies": preset_metadata.num_fields,
            "backend": config.backend,
            "adaptive": config.adaptive,
            "tolerance": config.tolerance if config.adaptive else None,
        },
        "simulation": {
            "totalFrames": len(frame_annotations),
            "numFrames": config.output.num_frames,
            "totalTime": total_time,
            "wallClockDuration": wall_clock_duration,
            "ndim": config.ndim,
            "resolution": config.resolution,
            "dtStatistics": dt_stats,
        },
        "visualization": _create_visualization_metadata(config, preset_metadata),
        "interventions": [],
        "frameAnnotations": frame_annotations,
    }

