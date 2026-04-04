"""Post-run diagnostics for exported simulation trajectories."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from plm_data.core.health import combine_health_status

DEFAULT_STAGNATION_REL_THRESHOLD = 1e-4
DEFAULT_STAGNATION_MIN_FRACTION = 0.2
DEFAULT_LOW_DYNAMIC_RANGE_REL_THRESHOLD = 1e-6
DEFAULT_LOW_DYNAMIC_RANGE_ABS_THRESHOLD = 1e-10
DEFAULT_SPATIAL_UNIFORMITY_REL_STD_THRESHOLD = 1e-6
DEFAULT_SPATIAL_UNIFORMITY_ABS_STD_THRESHOLD = 1e-10


def build_stagnation_report(
    field_frames: Mapping[str, Sequence[np.ndarray]],
    *,
    num_frames: int | None = None,
    skipped_static_fields: Sequence[str] = (),
    rel_threshold: float = DEFAULT_STAGNATION_REL_THRESHOLD,
    min_stagnant_fraction: float = DEFAULT_STAGNATION_MIN_FRACTION,
) -> dict[str, Any]:
    """Analyze exported field trajectories for trailing stagnation.

    The check operates on concrete exported outputs, e.g. ``u`` or
    ``velocity_x``. A field is considered stagnant when the trailing
    frame-to-frame change stays below ``rel_threshold`` for at least
    ``min_stagnant_fraction`` of the run.
    """

    checked_fields = list(field_frames)
    skipped_static = list(skipped_static_fields)
    inferred_num_frames = _frame_count(field_frames)
    if num_frames is None:
        num_frames = inferred_num_frames
    elif inferred_num_frames not in {0, num_frames}:
        raise ValueError(
            "Stagnation diagnostics received a frame count that does not match "
            "the stored field trajectories."
        )
    report: dict[str, Any] = {
        "applied": False,
        "reason": "insufficient_frames",
        "status": "pass",
        "rel_threshold": rel_threshold,
        "min_stagnant_fraction": min_stagnant_fraction,
        "checked_fields": checked_fields,
        "skipped_static_fields": skipped_static,
        "stagnant_fields": [],
        "fields": {},
    }

    if num_frames < 2:
        return report

    report["applied"] = True
    report["reason"] = None

    stagnant_fields: list[str] = []
    min_stagnant_count = max(int(min_stagnant_fraction * num_frames), 2)

    for field_name in checked_fields:
        frames = field_frames[field_name]
        field_scale = _field_scale(frames)

        if field_scale == 0.0:
            stagnant_fields.append(field_name)
            report["fields"][field_name] = {
                "stagnant": True,
                "stagnant_from_frame": 0,
                "trailing_stagnant_frames": num_frames,
                "field_range": 0.0,
                "max_relative_change": 0.0,
                "final_relative_change": 0.0,
            }
            continue

        relative_changes: list[float] = []
        for i in range(num_frames - 1):
            abs_diff = float(np.nanmax(np.abs(frames[i + 1] - frames[i])))
            relative_changes.append(abs_diff / field_scale)

        trailing_stagnant = 0
        for rel_change in reversed(relative_changes):
            if rel_change < rel_threshold:
                trailing_stagnant += 1
            else:
                break

        is_stagnant = trailing_stagnant >= min_stagnant_count
        stagnant_from_frame = (
            (num_frames - 1 - trailing_stagnant) if is_stagnant else None
        )
        if is_stagnant:
            stagnant_fields.append(field_name)

        report["fields"][field_name] = {
            "stagnant": is_stagnant,
            "stagnant_from_frame": stagnant_from_frame,
            "trailing_stagnant_frames": trailing_stagnant,
            "field_range": field_scale,
            "max_relative_change": max(relative_changes),
            "final_relative_change": relative_changes[-1],
        }

    report["stagnant_fields"] = stagnant_fields
    if stagnant_fields:
        report["status"] = "warn"
    return report


def build_finite_values_report(
    field_frames: Mapping[str, Sequence[np.ndarray]],
    *,
    num_frames: int | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check sampled output trajectories for in-domain NaN / Inf values."""

    checked_fields = list(field_frames)
    num_frames = _resolve_frame_count(
        field_frames,
        num_frames=num_frames,
        diagnostic_name="Finite-value diagnostics",
    )
    report: dict[str, Any] = {
        "applied": False,
        "reason": "insufficient_frames",
        "status": "pass",
        "checked_fields": checked_fields,
        "bad_fields": [],
        "fields": {},
    }
    if num_frames < 1:
        return report

    report["applied"] = True
    report["reason"] = None

    bad_fields: list[str] = []
    for field_name in checked_fields:
        bad_frames: list[int] = []
        bad_value_count = 0
        max_bad_fraction = 0.0
        valid_point_count = 0

        for frame_index, frame in enumerate(field_frames[field_name]):
            values = _valid_values(frame, valid_mask)
            valid_point_count = max(valid_point_count, int(values.size))
            if values.size == 0:
                continue

            bad_mask = ~np.isfinite(values)
            bad_count = int(np.count_nonzero(bad_mask))
            if bad_count == 0:
                continue

            bad_frames.append(frame_index)
            bad_value_count += bad_count
            max_bad_fraction = max(max_bad_fraction, bad_count / float(values.size))

        all_finite = not bad_frames
        if not all_finite:
            bad_fields.append(field_name)

        report["fields"][field_name] = {
            "all_finite": all_finite,
            "bad_frames": bad_frames,
            "bad_value_count": bad_value_count,
            "max_bad_fraction": max_bad_fraction,
            "valid_point_count": valid_point_count,
        }

    report["bad_fields"] = bad_fields
    if bad_fields:
        report["status"] = "fail"
    return report


def build_low_dynamic_range_report(
    field_frames: Mapping[str, Sequence[np.ndarray]],
    *,
    num_frames: int | None = None,
    valid_mask: np.ndarray | None = None,
    rel_threshold: float = DEFAULT_LOW_DYNAMIC_RANGE_REL_THRESHOLD,
    abs_threshold: float = DEFAULT_LOW_DYNAMIC_RANGE_ABS_THRESHOLD,
) -> dict[str, Any]:
    """Flag trajectories whose sampled range stays numerically tiny."""

    checked_fields = list(field_frames)
    num_frames = _resolve_frame_count(
        field_frames,
        num_frames=num_frames,
        diagnostic_name="Low-dynamic-range diagnostics",
    )
    report: dict[str, Any] = {
        "applied": False,
        "reason": "insufficient_frames",
        "status": "pass",
        "rel_threshold": rel_threshold,
        "abs_threshold": abs_threshold,
        "checked_fields": checked_fields,
        "low_dynamic_range_fields": [],
        "fields": {},
    }
    if num_frames < 1:
        return report

    report["applied"] = True
    report["reason"] = None

    low_dynamic_range_fields: list[str] = []
    for field_name in checked_fields:
        global_min = np.inf
        global_max = -np.inf
        max_abs = 0.0
        valid_point_count = 0

        for frame in field_frames[field_name]:
            values = _magnitude_values(frame, valid_mask)
            valid_point_count = max(valid_point_count, int(values.size))
            if values.size == 0:
                continue

            global_min = min(global_min, float(np.min(values)))
            global_max = max(global_max, float(np.max(values)))
            max_abs = max(max_abs, float(np.max(np.abs(values))))

        if valid_point_count == 0:
            field_range = 0.0
        else:
            field_range = global_max - global_min

        threshold = max(abs_threshold, rel_threshold * max(max_abs, 1.0))
        low_dynamic_range = field_range <= threshold
        if low_dynamic_range:
            low_dynamic_range_fields.append(field_name)

        report["fields"][field_name] = {
            "low_dynamic_range": low_dynamic_range,
            "field_range": field_range,
            "max_abs": max_abs,
            "threshold": threshold,
            "valid_point_count": valid_point_count,
        }

    report["low_dynamic_range_fields"] = low_dynamic_range_fields
    if low_dynamic_range_fields:
        report["status"] = "warn"
    return report


def build_spatial_uniformity_report(
    field_frames: Mapping[str, Sequence[np.ndarray]],
    *,
    num_frames: int | None = None,
    valid_mask: np.ndarray | None = None,
    rel_std_threshold: float = DEFAULT_SPATIAL_UNIFORMITY_REL_STD_THRESHOLD,
    abs_std_threshold: float = DEFAULT_SPATIAL_UNIFORMITY_ABS_STD_THRESHOLD,
) -> dict[str, Any]:
    """Flag trajectories whose sampled spatial variation stays tiny."""

    checked_fields = list(field_frames)
    num_frames = _resolve_frame_count(
        field_frames,
        num_frames=num_frames,
        diagnostic_name="Spatial-uniformity diagnostics",
    )
    report: dict[str, Any] = {
        "applied": False,
        "reason": "insufficient_frames",
        "status": "pass",
        "rel_std_threshold": rel_std_threshold,
        "abs_std_threshold": abs_std_threshold,
        "checked_fields": checked_fields,
        "spatially_uniform_fields": [],
        "fields": {},
    }
    if num_frames < 1:
        return report

    report["applied"] = True
    report["reason"] = None

    spatially_uniform_fields: list[str] = []
    for field_name in checked_fields:
        max_spatial_std = 0.0
        max_abs = 0.0
        valid_point_count = 0

        for frame in field_frames[field_name]:
            values = _magnitude_values(frame, valid_mask)
            valid_point_count = max(valid_point_count, int(values.size))
            if values.size == 0:
                continue

            max_spatial_std = max(max_spatial_std, float(np.std(values)))
            max_abs = max(max_abs, float(np.max(np.abs(values))))

        threshold = max(abs_std_threshold, rel_std_threshold * max(max_abs, 1.0))
        spatially_uniform = max_spatial_std <= threshold
        if spatially_uniform:
            spatially_uniform_fields.append(field_name)

        report["fields"][field_name] = {
            "spatially_uniform": spatially_uniform,
            "max_spatial_std": max_spatial_std,
            "max_abs": max_abs,
            "threshold": threshold,
            "valid_point_count": valid_point_count,
        }

    report["spatially_uniform_fields"] = spatially_uniform_fields
    if spatially_uniform_fields:
        report["status"] = "warn"
    return report


def build_output_health_report(
    field_frames: Mapping[str, Sequence[np.ndarray]],
    *,
    num_frames: int | None = None,
    skipped_static_fields: Sequence[str] = (),
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a combined post-run health report for sampled output trajectories."""

    checked_fields = list(field_frames)
    finite_values = build_finite_values_report(
        field_frames,
        num_frames=num_frames,
        valid_mask=valid_mask,
    )
    low_dynamic_range = build_low_dynamic_range_report(
        field_frames,
        num_frames=num_frames,
        valid_mask=valid_mask,
    )
    spatial_uniformity = build_spatial_uniformity_report(
        field_frames,
        num_frames=num_frames,
        valid_mask=valid_mask,
    )
    stagnation = build_stagnation_report(
        field_frames,
        num_frames=num_frames,
        skipped_static_fields=skipped_static_fields,
    )

    checks = {
        "finite_values": finite_values,
        "low_dynamic_range": low_dynamic_range,
        "spatial_uniformity": spatial_uniformity,
        "stagnation": stagnation,
    }
    statuses = [check["status"] for check in checks.values()]
    warning_checks = [
        name for name, check in checks.items() if check["status"] == "warn"
    ]
    failing_checks = [
        name for name, check in checks.items() if check["status"] == "fail"
    ]
    return {
        "applied": any(check["applied"] for check in checks.values()),
        "reason": None,
        "status": combine_health_status(*statuses),
        "checked_fields": checked_fields,
        "skipped_static_fields": list(skipped_static_fields),
        "warning_checks": warning_checks,
        "failing_checks": failing_checks,
        "checks": checks,
    }


def _frame_count(field_frames: Mapping[str, Sequence[np.ndarray]]) -> int:
    """Return the shared frame count for all tracked fields."""

    counts = {len(frames) for frames in field_frames.values()}
    if not counts:
        return 0
    if len(counts) != 1:
        raise ValueError(
            "Stagnation diagnostics require the same number of frames for all fields."
        )
    return counts.pop()


def _field_scale(frames: Sequence[np.ndarray]) -> float:
    """Return the normalization scale for one field trajectory."""

    if any(np.iscomplexobj(frame) for frame in frames):
        return max(float(np.nanmax(np.abs(frame))) for frame in frames)

    global_min = min(float(np.nanmin(frame)) for frame in frames)
    global_max = max(float(np.nanmax(frame)) for frame in frames)
    return global_max - global_min


def _resolve_frame_count(
    field_frames: Mapping[str, Sequence[np.ndarray]],
    *,
    num_frames: int | None,
    diagnostic_name: str,
) -> int:
    """Validate or infer the shared frame count for a diagnostic."""

    inferred_num_frames = _frame_count(field_frames)
    if num_frames is None:
        return inferred_num_frames
    if inferred_num_frames not in {0, num_frames}:
        raise ValueError(
            f"{diagnostic_name} received a frame count that does not match the "
            "stored field trajectories."
        )
    return num_frames


def _valid_values(frame: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    """Return the in-domain sample values for one output frame."""

    values = np.asarray(frame)
    if valid_mask is None:
        return values.reshape(-1)

    mask = np.asarray(valid_mask)
    if mask.shape != values.shape:
        raise ValueError(
            f"Diagnostic valid_mask shape {mask.shape} does not match frame shape "
            f"{values.shape}."
        )
    return values[mask]


def _magnitude_values(frame: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    """Return in-domain magnitudes for one frame, handling complex outputs."""

    values = _valid_values(frame, valid_mask)
    if np.iscomplexobj(values):
        return np.abs(values)
    return values
