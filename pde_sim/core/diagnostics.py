"""Trajectory stagnation detection for PDE simulations."""

from collections.abc import Callable
from typing import Any

import numpy as np


def check_trajectory_stagnation(
    all_fields: list,
    field_names: list[str],
    extract_field_fn: Callable,
    rel_threshold: float = 1e-4,
    min_stagnant_fraction: float = 0.2,
) -> dict[str, Any]:
    """Check if simulation fields have become stagnant (reached steady state).

    Computes frame-to-frame maximum absolute deviation per field, normalized
    by the field's total value range. If this relative change drops below a
    threshold for consecutive trailing frames, the field is flagged as stagnant.

    Args:
        all_fields: List of field states (one per saved frame).
        field_names: Names of fields to check.
        extract_field_fn: Callable(state, field_name) -> np.ndarray.
        rel_threshold: Relative change threshold below which a frame pair
            is considered stagnant.
        min_stagnant_fraction: Minimum fraction of total frames that must be
            stagnant at the trailing end to trigger a warning.

    Returns:
        Dictionary with stagnation analysis per field.
    """
    num_frames = len(all_fields)
    threshold_percent = rel_threshold * 100.0
    stagnant_fields: list[str] = []
    fields_info: dict[str, dict[str, Any]] = {}

    for field_name in field_names:
        # Extract numpy data for each frame
        frames_data = [extract_field_fn(state, field_name) for state in all_fields]

        # Compute overall field range across all frames
        global_min = float("inf")
        global_max = float("-inf")
        for data in frames_data:
            global_min = min(global_min, float(np.min(data)))
            global_max = max(global_max, float(np.max(data)))
        field_range = global_max - global_min

        # If range is 0, field is completely constant
        if field_range == 0:
            stagnant_fields.append(field_name)
            fields_info[field_name] = {
                "stagnant": True,
                "stagnant_from_frame": 0,
                "trailing_stagnant_frames": num_frames,
                "field_range": 0.0,
                "max_relative_change": 0.0,
                "final_relative_change": 0.0,
                "variability_percent": 0.0,
                "final_variability_percent": 0.0,
                "variability_below_threshold": True,
            }
            continue

        # Compute relative changes between consecutive frames
        relative_changes: list[float] = []
        for i in range(num_frames - 1):
            max_abs_diff = float(np.max(np.abs(frames_data[i + 1] - frames_data[i])))
            relative_changes.append(max_abs_diff / field_range)

        max_relative_change = max(relative_changes) if relative_changes else 0.0
        final_relative_change = relative_changes[-1] if relative_changes else 0.0

        # Count trailing stagnant streak from the end
        trailing_stagnant = 0
        for rc in reversed(relative_changes):
            if rc < rel_threshold:
                trailing_stagnant += 1
            else:
                break

        # Minimum stagnant frames required to trigger warning
        min_stagnant_count = max(int(min_stagnant_fraction * num_frames), 2)
        is_stagnant = trailing_stagnant >= min_stagnant_count

        # stagnant_from_frame: the frame index where stagnation begins
        # If trailing_stagnant == N, stagnation starts at frame (num_frames - 1 - N)
        # because relative_changes[i] compares frame i and i+1
        stagnant_from_frame = (num_frames - 1 - trailing_stagnant) if is_stagnant else None

        if is_stagnant:
            stagnant_fields.append(field_name)

        fields_info[field_name] = {
            "stagnant": is_stagnant,
            "stagnant_from_frame": stagnant_from_frame,
            "trailing_stagnant_frames": trailing_stagnant,
            "field_range": field_range,
            "max_relative_change": max_relative_change,
            "final_relative_change": final_relative_change,
            "variability_percent": max_relative_change * 100.0,
            "final_variability_percent": final_relative_change * 100.0,
            "variability_below_threshold": final_relative_change < rel_threshold,
        }

    return {
        "stagnant_fields": stagnant_fields,
        "relative_threshold": rel_threshold,
        "variability_threshold_percent": threshold_percent,
        "fields": fields_info,
    }
