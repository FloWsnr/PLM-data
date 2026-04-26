"""Random sampling profiles and fallback coordinate samplers for domains."""

from typing import Any

from plm_data.core.runtime_config import DomainConfig
from plm_data.domains.base import CoordinateSample
from plm_data.sampling.specs import RandomDomainProfile


def _uniform(context, name: str, minimum: float, maximum: float) -> float:
    from plm_data.sampling.samplers import uniform

    return uniform(context, name, minimum, maximum)


def _randint(context, name: str, minimum: int, maximum: int) -> int:
    from plm_data.sampling.samplers import randint

    return randint(context, name, minimum, maximum)


def _range(
    constraints: dict[str, Any],
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    raw = constraints.get(name)
    if raw is None:
        return default
    if not isinstance(raw, tuple | list) or len(raw) != 2:
        raise ValueError(f"Domain constraint '{name}' must be a two-value range.")
    return (float(raw[0]), float(raw[1]))


def _int_range(
    constraints: dict[str, Any],
    name: str,
    default: tuple[int, int],
) -> tuple[int, int]:
    raw = constraints.get(name)
    if raw is None:
        return default
    if not isinstance(raw, tuple | list) or len(raw) != 2:
        raise ValueError(f"Domain constraint '{name}' must be a two-value range.")
    return (int(raw[0]), int(raw[1]))


def _rectangle(context, constraints: dict[str, Any]) -> DomainConfig:
    length = _uniform(
        context, "domain.length", *_range(constraints, "length", (1.0, 2.0))
    )
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.8, 1.4))
    )
    cells_x = _randint(
        context, "domain.cells_x", *_int_range(constraints, "cells_x", (24, 40))
    )
    cells_y = max(8, int(round(cells_x * height / length)))
    return DomainConfig(
        type="rectangle",
        params={"size": [length, height], "mesh_resolution": [cells_x, cells_y]},
    )


def _disk(context, constraints: dict[str, Any]) -> DomainConfig:
    radius = _uniform(
        context, "domain.radius", *_range(constraints, "radius", (0.85, 1.15))
    )
    return DomainConfig(
        type="disk",
        params={
            "center": [0.0, 0.0],
            "radius": radius,
            "mesh_size": _uniform(
                context,
                "domain.mesh_size",
                *_range(constraints, "mesh_size", (0.055, 0.075)),
            ),
        },
    )


def _annulus(context, constraints: dict[str, Any]) -> DomainConfig:
    inner_radius = _uniform(
        context,
        "domain.inner_radius",
        *_range(constraints, "inner_radius", (0.22, 0.38)),
    )
    outer_radius = _uniform(
        context, "domain.outer_radius", *_range(constraints, "outer_radius", (0.9, 1.2))
    )
    return DomainConfig(
        type="annulus",
        params={
            "center": [0.0, 0.0],
            "inner_radius": inner_radius,
            "outer_radius": max(outer_radius, inner_radius + 0.45),
            "mesh_size": _uniform(
                context,
                "domain.mesh_size",
                *_range(constraints, "mesh_size", (0.055, 0.075)),
            ),
        },
    )


def _channel_obstacle(context, constraints: dict[str, Any]) -> DomainConfig:
    length = _uniform(
        context, "domain.length", *_range(constraints, "length", (1.8, 3.0))
    )
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.8, 1.4))
    )
    radius = _uniform(context, "domain.obstacle_radius", 0.07 * height, 0.14 * height)
    return DomainConfig(
        type="channel_obstacle",
        params={
            "length": length,
            "height": height,
            "obstacle_center": [
                _uniform(
                    context, "domain.obstacle_center_x", 0.35 * length, 0.55 * length
                ),
                _uniform(
                    context, "domain.obstacle_center_y", 0.4 * height, 0.6 * height
                ),
            ],
            "obstacle_radius": radius,
            "mesh_size": _uniform(context, "domain.mesh_size", 0.06, 0.1),
        },
    )


def _airfoil_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.2, 3.2))
    )
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.0, 1.4))
    )
    chord = _uniform(context, "domain.chord_length", 0.25 * length, 0.34 * length)
    return DomainConfig(
        type="airfoil_channel",
        params={
            "length": length,
            "height": height,
            "airfoil_center": [0.45 * length, 0.5 * height],
            "chord_length": chord,
            "thickness_ratio": _uniform(context, "domain.thickness_ratio", 0.09, 0.15),
            "attack_angle_degrees": _uniform(context, "domain.attack_angle", -8.0, 8.0),
            "mesh_size": _uniform(context, "domain.mesh_size", 0.06, 0.1),
        },
    )


def _dumbbell(context, constraints: dict[str, Any]) -> DomainConfig:
    radius = _uniform(
        context, "domain.lobe_radius", *_range(constraints, "lobe_radius", (0.35, 0.5))
    )
    separation = _uniform(context, "domain.separation", 2.4 * radius, 3.2 * radius)
    return DomainConfig(
        type="dumbbell",
        params={
            "left_center": [-0.5 * separation, 0.0],
            "right_center": [0.5 * separation, 0.0],
            "lobe_radius": radius,
            "neck_width": _uniform(
                context, "domain.neck_width", 0.5 * radius, 1.1 * radius
            ),
            "mesh_size": _uniform(context, "domain.mesh_size", 0.055, 0.09),
        },
    )


def _l_shape(context, constraints: dict[str, Any]) -> DomainConfig:
    width = _uniform(
        context, "domain.outer_width", *_range(constraints, "outer_width", (1.1, 1.8))
    )
    height = _uniform(
        context, "domain.outer_height", *_range(constraints, "outer_height", (1.1, 1.8))
    )
    return DomainConfig(
        type="l_shape",
        params={
            "outer_width": width,
            "outer_height": height,
            "cutout_width": _uniform(
                context, "domain.cutout_width", 0.25 * width, 0.48 * width
            ),
            "cutout_height": _uniform(
                context, "domain.cutout_height", 0.25 * height, 0.48 * height
            ),
            "mesh_size": _uniform(context, "domain.mesh_size", 0.055, 0.09),
        },
    )


def _multi_hole_plate(context, constraints: dict[str, Any]) -> DomainConfig:
    width = _uniform(context, "domain.width", *_range(constraints, "width", (1.8, 2.6)))
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.1, 1.7))
    )
    radius = _uniform(
        context,
        "domain.hole_radius",
        0.07 * min(width, height),
        0.11 * min(width, height),
    )
    return DomainConfig(
        type="multi_hole_plate",
        params={
            "width": width,
            "height": height,
            "holes": [
                {"center": [0.35 * width, 0.5 * height], "radius": radius},
                {"center": [0.65 * width, 0.5 * height], "radius": radius},
            ],
            "mesh_size": _uniform(context, "domain.mesh_size", 0.055, 0.09),
        },
    )


def _parallelogram(context, constraints: dict[str, Any]) -> DomainConfig:
    length = _uniform(
        context, "domain.length", *_range(constraints, "length", (1.0, 1.8))
    )
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.8, 1.4))
    )
    skew_x = _uniform(context, "domain.skew_x", -0.2 * height, 0.2 * height)
    skew_y = _uniform(context, "domain.skew_y", -0.15 * length, 0.15 * length)
    cells_x = _randint(
        context, "domain.cells_x", *_int_range(constraints, "cells_x", (24, 38))
    )
    cells_y = max(8, int(round(cells_x * height / length)))
    return DomainConfig(
        type="parallelogram",
        params={
            "origin": [0.0, 0.0],
            "axis_x": [length, skew_x],
            "axis_y": [skew_y, height],
            "mesh_resolution": [cells_x, cells_y],
        },
    )


def _porous_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.2, 3.2))
    )
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.0, 1.5))
    )
    n_rows = _randint(context, "domain.n_rows", 2, 3)
    n_cols = _randint(context, "domain.n_cols", 3, 4)
    radius = _uniform(context, "domain.obstacle_radius", 0.035 * height, 0.055 * height)
    x_margin = 0.2 * length
    y_margin = 0.2 * height
    pitch_x = (length - 2.0 * x_margin - 2.0 * radius) / max(1, n_cols - 1)
    pitch_y = (height - 2.0 * y_margin - 2.0 * radius) / max(1, n_rows - 1)
    return DomainConfig(
        type="porous_channel",
        params={
            "length": length,
            "height": height,
            "obstacle_radius": radius,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "pitch_x": pitch_x,
            "pitch_y": pitch_y,
            "x_margin": x_margin,
            "y_margin": y_margin,
            "row_shift_fraction": _uniform(
                context, "domain.row_shift_fraction", 0.0, 0.3
            ),
            "mesh_size": _uniform(context, "domain.mesh_size", 0.055, 0.09),
        },
    )


def _serpentine_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    width = _uniform(
        context,
        "domain.channel_width",
        *_range(constraints, "channel_width", (0.12, 0.2)),
    )
    return DomainConfig(
        type="serpentine_channel",
        params={
            "channel_length": _uniform(context, "domain.channel_length", 1.0, 1.8),
            "lane_spacing": _uniform(
                context, "domain.lane_spacing", 1.8 * width, 3.0 * width
            ),
            "n_bends": _randint(context, "domain.n_bends", 2, 4),
            "channel_width": width,
            "mesh_size": _uniform(context, "domain.mesh_size", 0.045, 0.075),
        },
    )


def _side_cavity_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.0, 3.0))
    )
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.7, 1.1))
    )
    cavity_width = _uniform(
        context, "domain.cavity_width", 0.18 * length, 0.28 * length
    )
    return DomainConfig(
        type="side_cavity_channel",
        params={
            "length": length,
            "height": height,
            "cavity_width": cavity_width,
            "cavity_depth": _uniform(
                context, "domain.cavity_depth", 0.25 * height, 0.55 * height
            ),
            "cavity_center_x": _uniform(
                context, "domain.cavity_center_x", 0.35 * length, 0.65 * length
            ),
            "mesh_size": _uniform(context, "domain.mesh_size", 0.055, 0.09),
        },
    )


def _venturi_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.2, 3.2))
    )
    height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.0, 1.5))
    )
    throat_height = _uniform(
        context, "domain.throat_height", 0.45 * height, 0.7 * height
    )
    indentation = 0.5 * (height - throat_height)
    radius = _uniform(
        context, "domain.constriction_radius", 1.6 * indentation, 2.5 * indentation
    )
    return DomainConfig(
        type="venturi_channel",
        params={
            "length": length,
            "height": height,
            "throat_height": throat_height,
            "constriction_center_x": 0.5 * length,
            "constriction_radius": radius,
            "mesh_size": _uniform(context, "domain.mesh_size", 0.055, 0.09),
        },
    )


def _y_bifurcation(context, constraints: dict[str, Any]) -> DomainConfig:
    width = _uniform(
        context,
        "domain.channel_width",
        *_range(constraints, "channel_width", (0.16, 0.26)),
    )
    return DomainConfig(
        type="y_bifurcation",
        params={
            "inlet_length": _uniform(context, "domain.inlet_length", 0.9, 1.4),
            "branch_length": _uniform(context, "domain.branch_length", 1.0, 1.6),
            "branch_angle_degrees": _uniform(
                context, "domain.branch_angle", 35.0, 55.0
            ),
            "channel_width": width,
            "mesh_size": _uniform(context, "domain.mesh_size", 0.045, 0.075),
        },
    )


_DOMAIN_SAMPLERS = {
    "airfoil_channel": _airfoil_channel,
    "annulus": _annulus,
    "channel_obstacle": _channel_obstacle,
    "disk": _disk,
    "dumbbell": _dumbbell,
    "l_shape": _l_shape,
    "multi_hole_plate": _multi_hole_plate,
    "parallelogram": _parallelogram,
    "porous_channel": _porous_channel,
    "rectangle": _rectangle,
    "serpentine_channel": _serpentine_channel,
    "side_cavity_channel": _side_cavity_channel,
    "venturi_channel": _venturi_channel,
    "y_bifurcation": _y_bifurcation,
}


def random_profiles_for_domain(domain_name: str) -> tuple[RandomDomainProfile, ...]:
    """Return default random profiles for a registered domain spec."""
    if domain_name not in _DOMAIN_SAMPLERS:
        return ()
    return (
        RandomDomainProfile(
            name="default",
            sample=_DOMAIN_SAMPLERS[domain_name],
            description=f"Default random profile for {domain_name}.",
        ),
    )


def fallback_coordinate_sample(context, domain, region: str) -> CoordinateSample:
    """Return a conservative coordinate sample for domains without local samplers."""
    p = domain.params
    if domain.type in {
        "airfoil_channel",
        "channel_obstacle",
        "porous_channel",
        "side_cavity_channel",
        "venturi_channel",
    }:
        length = float(p["length"])
        height = float(p["height"])
        return CoordinateSample(
            point=[
                _uniform(
                    context, f"domain.region.{region}.x", 0.22 * length, 0.35 * length
                ),
                _uniform(
                    context, f"domain.region.{region}.y", 0.4 * height, 0.6 * height
                ),
            ],
            scale=min(length, height),
        )
    if domain.type == "dumbbell":
        center = (
            p["left_center"]
            if region in {"interior", "left_lobe"}
            else p["right_center"]
        )
        return CoordinateSample(
            point=[float(center[0]), float(center[1])], scale=float(p["lobe_radius"])
        )
    if domain.type == "l_shape":
        width = float(p["outer_width"])
        height = float(p["outer_height"])
        return CoordinateSample(
            point=[0.25 * width, 0.25 * height], scale=min(width, height)
        )
    if domain.type == "multi_hole_plate":
        width = float(p["width"])
        height = float(p["height"])
        return CoordinateSample(
            point=[0.2 * width, 0.2 * height], scale=min(width, height)
        )
    if domain.type == "parallelogram":
        origin = p["origin"]
        axis_x = p["axis_x"]
        axis_y = p["axis_y"]
        return CoordinateSample(
            point=[
                float(origin[0]) + 0.5 * float(axis_x[0]) + 0.5 * float(axis_y[0]),
                float(origin[1]) + 0.5 * float(axis_x[1]) + 0.5 * float(axis_y[1]),
            ],
            scale=min(float(abs(axis_x[0])), float(abs(axis_y[1]))),
        )
    if domain.type == "serpentine_channel":
        return CoordinateSample(
            point=[0.5 * float(p["channel_length"]), 0.0],
            scale=float(p["channel_width"]),
        )
    if domain.type == "y_bifurcation":
        return CoordinateSample(
            point=[0.5 * float(p["inlet_length"]), 0.0], scale=float(p["channel_width"])
        )
    raise ValueError(
        f"Domain '{domain.type}' does not expose a coordinate-region sampler for "
        f"'{region}'."
    )
