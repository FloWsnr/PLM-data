"""Post-run diagnostics for exported simulation trajectories."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_STAGNATION_REL_THRESHOLD = 1e-4
DEFAULT_STAGNATION_MIN_FRACTION = 0.2


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
    return report


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
