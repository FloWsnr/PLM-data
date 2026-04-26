"""Random sampling profiles and fallback coordinate samplers for domains."""

import math
from typing import Any

from plm_data.core.runtime_config import DomainConfig
from plm_data.domains.base import CoordinateSample
from plm_data.sampling.specs import RandomDomainProfile

UNIT_DOMAIN_TOLERANCE = 1.0e-12


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


def _domain_extent(bounds: tuple[float, float, float, float]) -> float:
    x_min, x_max, y_min, y_max = bounds
    return max(x_max - x_min, y_max - y_min)


def _require_positive_scale(scale: float, domain_type: str) -> float:
    if scale <= 0.0:
        raise ValueError(
            f"Cannot unit-normalize domain '{domain_type}' with non-positive "
            f"extent {scale!r}."
        )
    return scale


def _normalized(context, name: str, raw: float, scale: float) -> float:
    value = raw / scale
    context.values[f"{name}.raw"] = raw
    context.values[name] = value
    return value


def _normalized_mesh_size(context, raw: float, scale: float) -> float:
    return _normalized(context, "domain.mesh_size", raw, scale)


def random_domain_bounds(domain: DomainConfig) -> tuple[float, float, float, float]:
    """Return conservative x/y bounds for a sampled built-in random domain."""
    p = domain.params
    if domain.type == "rectangle":
        length, height = p["size"]
        return 0.0, float(length), 0.0, float(height)
    if domain.type == "disk":
        center = p["center"]
        radius = float(p["radius"])
        return (
            float(center[0]) - radius,
            float(center[0]) + radius,
            float(center[1]) - radius,
            float(center[1]) + radius,
        )
    if domain.type == "annulus":
        center = p["center"]
        outer_radius = float(p["outer_radius"])
        return (
            float(center[0]) - outer_radius,
            float(center[0]) + outer_radius,
            float(center[1]) - outer_radius,
            float(center[1]) + outer_radius,
        )
    if domain.type in {
        "airfoil_channel",
        "channel_obstacle",
        "porous_channel",
        "venturi_channel",
    }:
        return 0.0, float(p["length"]), 0.0, float(p["height"])
    if domain.type == "side_cavity_channel":
        return (
            0.0,
            float(p["length"]),
            0.0,
            float(p["height"]) + float(p["cavity_depth"]),
        )
    if domain.type == "dumbbell":
        left_center = p["left_center"]
        right_center = p["right_center"]
        radius = float(p["lobe_radius"])
        return (
            float(left_center[0]) - radius,
            float(right_center[0]) + radius,
            float(left_center[1]) - radius,
            float(left_center[1]) + radius,
        )
    if domain.type == "l_shape":
        return 0.0, float(p["outer_width"]), 0.0, float(p["outer_height"])
    if domain.type == "multi_hole_plate":
        return 0.0, float(p["width"]), 0.0, float(p["height"])
    if domain.type == "parallelogram":
        origin = p["origin"]
        axis_x = p["axis_x"]
        axis_y = p["axis_y"]
        points = (
            (float(origin[0]), float(origin[1])),
            (float(origin[0]) + float(axis_x[0]), float(origin[1]) + float(axis_x[1])),
            (
                float(origin[0]) + float(axis_x[0]) + float(axis_y[0]),
                float(origin[1]) + float(axis_x[1]) + float(axis_y[1]),
            ),
            (float(origin[0]) + float(axis_y[0]), float(origin[1]) + float(axis_y[1])),
        )
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), max(xs), min(ys), max(ys)
    if domain.type == "serpentine_channel":
        half_width = 0.5 * float(p["channel_width"])
        return (
            -half_width,
            float(p["channel_length"]) + half_width,
            -half_width,
            int(p["n_bends"]) * float(p["lane_spacing"]) + half_width,
        )
    if domain.type == "y_bifurcation":
        inlet_length = float(p["inlet_length"])
        branch_length = float(p["branch_length"])
        angle = math.radians(float(p["branch_angle_degrees"]))
        half_width = 0.5 * float(p["channel_width"])

        def _rotated_branch_corners(sign: float) -> tuple[tuple[float, float], ...]:
            signed_angle = sign * angle
            cos_a = math.cos(signed_angle)
            sin_a = math.sin(signed_angle)
            corners = (
                (0.0, -half_width),
                (branch_length, -half_width),
                (branch_length, half_width),
                (0.0, half_width),
            )
            return tuple(
                (
                    inlet_length + x_coord * cos_a - y_coord * sin_a,
                    x_coord * sin_a + y_coord * cos_a,
                )
                for x_coord, y_coord in corners
            )

        points = (
            (0.0, -half_width),
            (inlet_length, -half_width),
            (inlet_length, half_width),
            (0.0, half_width),
            *_rotated_branch_corners(1.0),
            *_rotated_branch_corners(-1.0),
        )
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), max(xs), min(ys), max(ys)
    raise ValueError(f"Unsupported random domain bounds for '{domain.type}'.")


def random_domain_extent(domain: DomainConfig) -> float:
    """Return the maximum x/y extent of one sampled random domain."""
    return _domain_extent(random_domain_bounds(domain))


def validate_unit_random_domain(domain: DomainConfig) -> None:
    """Validate that a sampled random domain has unit maximum extent."""
    extent = random_domain_extent(domain)
    if not math.isclose(
        extent,
        1.0,
        rel_tol=0.0,
        abs_tol=UNIT_DOMAIN_TOLERANCE,
    ):
        raise ValueError(
            f"Random domain '{domain.type}' must have maximum extent 1. Got {extent!r}."
        )


def _rectangle(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_length = _uniform(
        context, "domain.length", *_range(constraints, "length", (1.0, 2.0))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.8, 1.4))
    )
    scale = _require_positive_scale(max(raw_length, raw_height), "rectangle")
    length = _normalized(context, "domain.length", raw_length, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    context.values["domain.unit_scale"] = scale
    cells_x = _randint(
        context, "domain.cells_x", *_int_range(constraints, "cells_x", (24, 40))
    )
    cells_y = max(8, int(round(cells_x * height / length)))
    return DomainConfig(
        type="rectangle",
        params={"size": [length, height], "mesh_resolution": [cells_x, cells_y]},
    )


def _disk(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_radius = _uniform(
        context, "domain.radius", *_range(constraints, "radius", (0.85, 1.15))
    )
    raw_mesh_size = _uniform(
        context,
        "domain.mesh_size",
        *_range(constraints, "mesh_size", (0.055, 0.075)),
    )
    scale = _require_positive_scale(2.0 * raw_radius, "disk")
    radius = _normalized(context, "domain.radius", raw_radius, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="disk",
        params={
            "center": [0.5, 0.5],
            "radius": radius,
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _annulus(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_inner_radius = _uniform(
        context,
        "domain.inner_radius",
        *_range(constraints, "inner_radius", (0.22, 0.38)),
    )
    sampled_outer_radius = _uniform(
        context, "domain.outer_radius", *_range(constraints, "outer_radius", (0.9, 1.2))
    )
    raw_outer_radius = max(sampled_outer_radius, raw_inner_radius + 0.45)
    raw_mesh_size = _uniform(
        context,
        "domain.mesh_size",
        *_range(constraints, "mesh_size", (0.055, 0.075)),
    )
    scale = _require_positive_scale(2.0 * raw_outer_radius, "annulus")
    inner_radius = _normalized(context, "domain.inner_radius", raw_inner_radius, scale)
    outer_radius = _normalized(context, "domain.outer_radius", raw_outer_radius, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="annulus",
        params={
            "center": [0.5, 0.5],
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _channel_obstacle(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_length = _uniform(
        context, "domain.length", *_range(constraints, "length", (1.8, 3.0))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.8, 1.4))
    )
    raw_radius = _uniform(
        context, "domain.obstacle_radius", 0.07 * raw_height, 0.14 * raw_height
    )
    raw_center_x = _uniform(
        context, "domain.obstacle_center_x", 0.35 * raw_length, 0.55 * raw_length
    )
    raw_center_y = _uniform(
        context, "domain.obstacle_center_y", 0.4 * raw_height, 0.6 * raw_height
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.06, 0.1)
    scale = _require_positive_scale(max(raw_length, raw_height), "channel_obstacle")
    length = _normalized(context, "domain.length", raw_length, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="channel_obstacle",
        params={
            "length": length,
            "height": height,
            "obstacle_center": [
                _normalized(context, "domain.obstacle_center_x", raw_center_x, scale),
                _normalized(context, "domain.obstacle_center_y", raw_center_y, scale),
            ],
            "obstacle_radius": _normalized(
                context, "domain.obstacle_radius", raw_radius, scale
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _airfoil_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.2, 3.2))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.0, 1.4))
    )
    raw_chord = _uniform(
        context, "domain.chord_length", 0.25 * raw_length, 0.34 * raw_length
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.06, 0.1)
    scale = _require_positive_scale(max(raw_length, raw_height), "airfoil_channel")
    length = _normalized(context, "domain.length", raw_length, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="airfoil_channel",
        params={
            "length": length,
            "height": height,
            "airfoil_center": [0.45 * length, 0.5 * height],
            "chord_length": _normalized(
                context, "domain.chord_length", raw_chord, scale
            ),
            "thickness_ratio": _uniform(context, "domain.thickness_ratio", 0.09, 0.15),
            "attack_angle_degrees": _uniform(context, "domain.attack_angle", -8.0, 8.0),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _dumbbell(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_radius = _uniform(
        context, "domain.lobe_radius", *_range(constraints, "lobe_radius", (0.35, 0.5))
    )
    raw_separation = _uniform(
        context, "domain.separation", 2.4 * raw_radius, 3.2 * raw_radius
    )
    raw_neck_width = _uniform(
        context, "domain.neck_width", 0.5 * raw_radius, 1.1 * raw_radius
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.055, 0.09)
    scale = _require_positive_scale(raw_separation + 2.0 * raw_radius, "dumbbell")
    radius = _normalized(context, "domain.lobe_radius", raw_radius, scale)
    separation = _normalized(context, "domain.separation", raw_separation, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="dumbbell",
        params={
            "left_center": [radius, 0.5],
            "right_center": [radius + separation, 0.5],
            "lobe_radius": radius,
            "neck_width": _normalized(
                context, "domain.neck_width", raw_neck_width, scale
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _l_shape(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_width = _uniform(
        context, "domain.outer_width", *_range(constraints, "outer_width", (1.1, 1.8))
    )
    raw_height = _uniform(
        context, "domain.outer_height", *_range(constraints, "outer_height", (1.1, 1.8))
    )
    raw_cutout_width = _uniform(
        context, "domain.cutout_width", 0.25 * raw_width, 0.48 * raw_width
    )
    raw_cutout_height = _uniform(
        context, "domain.cutout_height", 0.25 * raw_height, 0.48 * raw_height
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.055, 0.09)
    scale = _require_positive_scale(max(raw_width, raw_height), "l_shape")
    width = _normalized(context, "domain.outer_width", raw_width, scale)
    height = _normalized(context, "domain.outer_height", raw_height, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="l_shape",
        params={
            "outer_width": width,
            "outer_height": height,
            "cutout_width": _normalized(
                context, "domain.cutout_width", raw_cutout_width, scale
            ),
            "cutout_height": _normalized(
                context, "domain.cutout_height", raw_cutout_height, scale
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _multi_hole_plate(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_width = _uniform(
        context, "domain.width", *_range(constraints, "width", (1.8, 2.6))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.1, 1.7))
    )
    raw_radius = _uniform(
        context,
        "domain.hole_radius",
        0.07 * min(raw_width, raw_height),
        0.11 * min(raw_width, raw_height),
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.055, 0.09)
    scale = _require_positive_scale(max(raw_width, raw_height), "multi_hole_plate")
    width = _normalized(context, "domain.width", raw_width, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    radius = _normalized(context, "domain.hole_radius", raw_radius, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="multi_hole_plate",
        params={
            "width": width,
            "height": height,
            "holes": [
                {"center": [0.35 * width, 0.5 * height], "radius": radius},
                {"center": [0.65 * width, 0.5 * height], "radius": radius},
            ],
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _parallelogram(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_length = _uniform(
        context, "domain.length", *_range(constraints, "length", (1.0, 1.8))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.8, 1.4))
    )
    raw_skew_x = _uniform(context, "domain.skew_x", -0.2 * raw_height, 0.2 * raw_height)
    raw_skew_y = _uniform(
        context, "domain.skew_y", -0.15 * raw_length, 0.15 * raw_length
    )
    raw_domain = DomainConfig(
        type="parallelogram",
        params={
            "origin": [0.0, 0.0],
            "axis_x": [raw_length, raw_skew_x],
            "axis_y": [raw_skew_y, raw_height],
            "mesh_resolution": [1, 1],
        },
    )
    raw_bounds = random_domain_bounds(raw_domain)
    scale = _require_positive_scale(_domain_extent(raw_bounds), "parallelogram")
    origin = [-raw_bounds[0] / scale, -raw_bounds[2] / scale]
    length = _normalized(context, "domain.length", raw_length, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    skew_x = _normalized(context, "domain.skew_x", raw_skew_x, scale)
    skew_y = _normalized(context, "domain.skew_y", raw_skew_y, scale)
    context.values["domain.unit_scale"] = scale
    cells_x = _randint(
        context, "domain.cells_x", *_int_range(constraints, "cells_x", (24, 38))
    )
    cells_y = max(8, int(round(cells_x * height / length)))
    return DomainConfig(
        type="parallelogram",
        params={
            "origin": origin,
            "axis_x": [length, skew_x],
            "axis_y": [skew_y, height],
            "mesh_resolution": [cells_x, cells_y],
        },
    )


def _porous_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.2, 3.2))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.0, 1.5))
    )
    n_rows = _randint(context, "domain.n_rows", 2, 3)
    n_cols = _randint(context, "domain.n_cols", 3, 4)
    raw_radius = _uniform(
        context, "domain.obstacle_radius", 0.035 * raw_height, 0.055 * raw_height
    )
    raw_x_margin = 0.2 * raw_length
    raw_y_margin = 0.2 * raw_height
    raw_pitch_x = (raw_length - 2.0 * raw_x_margin - 2.0 * raw_radius) / max(
        1, n_cols - 1
    )
    raw_pitch_y = (raw_height - 2.0 * raw_y_margin - 2.0 * raw_radius) / max(
        1, n_rows - 1
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.055, 0.09)
    scale = _require_positive_scale(max(raw_length, raw_height), "porous_channel")
    length = _normalized(context, "domain.length", raw_length, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="porous_channel",
        params={
            "length": length,
            "height": height,
            "obstacle_radius": _normalized(
                context, "domain.obstacle_radius", raw_radius, scale
            ),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "pitch_x": _normalized(context, "domain.pitch_x", raw_pitch_x, scale),
            "pitch_y": _normalized(context, "domain.pitch_y", raw_pitch_y, scale),
            "x_margin": _normalized(context, "domain.x_margin", raw_x_margin, scale),
            "y_margin": _normalized(context, "domain.y_margin", raw_y_margin, scale),
            "row_shift_fraction": _uniform(
                context, "domain.row_shift_fraction", 0.0, 0.3
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _serpentine_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_width = _uniform(
        context,
        "domain.channel_width",
        *_range(constraints, "channel_width", (0.12, 0.2)),
    )
    raw_channel_length = _uniform(context, "domain.channel_length", 1.0, 1.8)
    raw_lane_spacing = _uniform(
        context, "domain.lane_spacing", 1.8 * raw_width, 3.0 * raw_width
    )
    n_bends = _randint(context, "domain.n_bends", 2, 4)
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.045, 0.075)
    raw_domain = DomainConfig(
        type="serpentine_channel",
        params={
            "channel_length": raw_channel_length,
            "lane_spacing": raw_lane_spacing,
            "n_bends": n_bends,
            "channel_width": raw_width,
            "mesh_size": raw_mesh_size,
        },
    )
    scale = _require_positive_scale(
        random_domain_extent(raw_domain), "serpentine_channel"
    )
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="serpentine_channel",
        params={
            "channel_length": _normalized(
                context, "domain.channel_length", raw_channel_length, scale
            ),
            "lane_spacing": _normalized(
                context, "domain.lane_spacing", raw_lane_spacing, scale
            ),
            "n_bends": n_bends,
            "channel_width": _normalized(
                context, "domain.channel_width", raw_width, scale
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _side_cavity_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.0, 3.0))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (0.7, 1.1))
    )
    raw_cavity_width = _uniform(
        context, "domain.cavity_width", 0.18 * raw_length, 0.28 * raw_length
    )
    raw_cavity_depth = _uniform(
        context, "domain.cavity_depth", 0.25 * raw_height, 0.55 * raw_height
    )
    raw_cavity_center_x = _uniform(
        context, "domain.cavity_center_x", 0.35 * raw_length, 0.65 * raw_length
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.055, 0.09)
    scale = _require_positive_scale(
        max(raw_length, raw_height + raw_cavity_depth), "side_cavity_channel"
    )
    length = _normalized(context, "domain.length", raw_length, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="side_cavity_channel",
        params={
            "length": length,
            "height": height,
            "cavity_width": _normalized(
                context, "domain.cavity_width", raw_cavity_width, scale
            ),
            "cavity_depth": _normalized(
                context, "domain.cavity_depth", raw_cavity_depth, scale
            ),
            "cavity_center_x": _normalized(
                context, "domain.cavity_center_x", raw_cavity_center_x, scale
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _venturi_channel(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_length = _uniform(
        context, "domain.length", *_range(constraints, "length", (2.2, 3.2))
    )
    raw_height = _uniform(
        context, "domain.height", *_range(constraints, "height", (1.0, 1.5))
    )
    raw_throat_height = _uniform(
        context, "domain.throat_height", 0.45 * raw_height, 0.7 * raw_height
    )
    indentation = 0.5 * (raw_height - raw_throat_height)
    raw_radius = _uniform(
        context,
        "domain.constriction_radius",
        1.6 * indentation,
        2.5 * indentation,
    )
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.055, 0.09)
    scale = _require_positive_scale(max(raw_length, raw_height), "venturi_channel")
    length = _normalized(context, "domain.length", raw_length, scale)
    height = _normalized(context, "domain.height", raw_height, scale)
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="venturi_channel",
        params={
            "length": length,
            "height": height,
            "throat_height": _normalized(
                context, "domain.throat_height", raw_throat_height, scale
            ),
            "constriction_center_x": 0.5 * length,
            "constriction_radius": _normalized(
                context, "domain.constriction_radius", raw_radius, scale
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
        },
    )


def _y_bifurcation(context, constraints: dict[str, Any]) -> DomainConfig:
    raw_width = _uniform(
        context,
        "domain.channel_width",
        *_range(constraints, "channel_width", (0.16, 0.26)),
    )
    raw_inlet_length = _uniform(context, "domain.inlet_length", 0.9, 1.4)
    raw_branch_length = _uniform(context, "domain.branch_length", 1.0, 1.6)
    branch_angle = _uniform(context, "domain.branch_angle", 35.0, 55.0)
    raw_mesh_size = _uniform(context, "domain.mesh_size", 0.045, 0.075)
    raw_domain = DomainConfig(
        type="y_bifurcation",
        params={
            "inlet_length": raw_inlet_length,
            "branch_length": raw_branch_length,
            "branch_angle_degrees": branch_angle,
            "channel_width": raw_width,
            "mesh_size": raw_mesh_size,
        },
    )
    scale = _require_positive_scale(random_domain_extent(raw_domain), "y_bifurcation")
    context.values["domain.unit_scale"] = scale
    return DomainConfig(
        type="y_bifurcation",
        params={
            "inlet_length": _normalized(
                context, "domain.inlet_length", raw_inlet_length, scale
            ),
            "branch_length": _normalized(
                context, "domain.branch_length", raw_branch_length, scale
            ),
            "branch_angle_degrees": branch_angle,
            "channel_width": _normalized(
                context, "domain.channel_width", raw_width, scale
            ),
            "mesh_size": _normalized_mesh_size(context, raw_mesh_size, scale),
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
