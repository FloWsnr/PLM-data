"""Health diagnostics for OpenFOAM-backed preset runs."""

from collections.abc import Collection, Iterable, Mapping
import re

import numpy as np

from plm_data.core.health import RuntimeHealthTracker, SolverHealthTracker

_FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TIME_RE = re.compile(r"^Time = (?P<value>.+)$")
_SOLVER_RE = re.compile(
    rf"^(?P<solver_type>[^:]+):\s+Solving for "
    rf"(?P<field>[^,]+),\s+Initial residual = (?P<initial>{_FLOAT_PATTERN}),\s+"
    rf"Final residual = (?P<final>{_FLOAT_PATTERN}),\s+No Iterations "
    rf"(?P<iterations>\d+)$"
)
_COURANT_RE = re.compile(
    rf"^(?P<label>.*?Courant Number)\s+mean:\s*(?P<mean>{_FLOAT_PATTERN})\s+max:\s*"
    rf"(?P<max>{_FLOAT_PATTERN})$"
)
_CONTINUITY_RE = re.compile(
    rf"continuity errors.*sum local = (?P<sum_local>{_FLOAT_PATTERN}),\s+global = "
    rf"(?P<global>{_FLOAT_PATTERN})(?:,\s+cumulative = "
    rf"(?P<cumulative>{_FLOAT_PATTERN}))?"
)
_OUTER_LOOP_NONCONVERGED_RE = re.compile(
    r"^(?P<solver_type>PIMPLE|PISO|SIMPLE):.*not converged.*$",
    flags=re.IGNORECASE,
)


def ingest_openfoam_log_health(
    log_lines: Iterable[str],
    *,
    solver_tracker: SolverHealthTracker,
    runtime_tracker: RuntimeHealthTracker,
) -> None:
    """Parse one OpenFOAM solver log into shared solver/runtime health trackers."""

    context = "initialization"
    runtime_metrics: dict[str, float] = {}

    def _flush_runtime_metrics() -> None:
        nonlocal runtime_metrics
        if runtime_metrics:
            runtime_tracker.record(context, runtime_metrics)
            runtime_metrics = {}

    for raw_line in log_lines:
        line = raw_line.strip()
        if not line or line.startswith("$ "):
            continue

        time_match = _TIME_RE.match(line)
        if time_match is not None:
            _flush_runtime_metrics()
            context = _time_context(time_match.group("value"))
            continue

        solver_match = _SOLVER_RE.match(line)
        if solver_match is not None:
            solver_tracker.record_external_observation(
                name=solver_match.group("field").strip(),
                context=context,
                kind="openfoam",
                solver_type=solver_match.group("solver_type").strip(),
                converged_reason=1,
                iterations=int(solver_match.group("iterations")),
                linear_iterations=int(solver_match.group("iterations")),
                residual_norm=float(solver_match.group("final")),
                initial_residual=float(solver_match.group("initial")),
            )
            continue

        if _OUTER_LOOP_NONCONVERGED_RE.match(line) is not None:
            outer_solver = line.split(":", maxsplit=1)[0].strip()
            solver_tracker.record_external_observation(
                name=outer_solver,
                context=context,
                kind="openfoam_outer_loop",
                solver_type=outer_solver,
                converged_reason=-1,
            )
            continue

        courant_match = _COURANT_RE.match(line)
        if courant_match is not None:
            prefix = _normalise_metric_prefix(courant_match.group("label"))
            runtime_metrics[f"{prefix}_mean"] = float(courant_match.group("mean"))
            runtime_metrics[f"{prefix}_max"] = float(courant_match.group("max"))
            continue

        continuity_match = _CONTINUITY_RE.search(line)
        if continuity_match is not None:
            runtime_metrics["continuity_sum_local"] = float(
                continuity_match.group("sum_local")
            )
            runtime_metrics["continuity_global_abs"] = abs(
                float(continuity_match.group("global"))
            )
            cumulative = continuity_match.group("cumulative")
            if cumulative is not None:
                runtime_metrics["continuity_cumulative_abs"] = abs(float(cumulative))

    _flush_runtime_metrics()


def sampled_field_runtime_metrics(
    sampled_fields: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
    *,
    positive_fields: Collection[str] = (),
    context: str,
) -> dict[str, float]:
    """Build runtime-health metrics from one sampled OpenFOAM output frame."""

    mask = np.asarray(valid_mask, dtype=bool).reshape(-1, order="C")
    if mask.size == 0 or not np.any(mask):
        raise ValueError(
            f"OpenFOAM sampling produced no valid in-domain points during {context}."
        )

    required_positive = set(positive_fields)
    metrics: dict[str, float] = {
        "sample_valid_fraction": float(np.mean(mask)),
    }
    for name, values in sampled_fields.items():
        flat_values = np.asarray(values, dtype=float).reshape(-1, order="C")
        if flat_values.size != mask.size:
            raise ValueError(
                f"OpenFOAM sampled field '{name}' has {flat_values.size} entries, "
                f"but the validity mask has {mask.size}."
            )

        in_domain_values = flat_values[mask]
        if not np.all(np.isfinite(in_domain_values)):
            raise ValueError(
                f"OpenFOAM output '{name}' contains non-finite in-domain values "
                f"during {context}."
            )

        min_value = float(np.min(in_domain_values))
        metrics[f"{name}_min"] = min_value
        metrics[f"{name}_max"] = float(np.max(in_domain_values))
        metrics[f"{name}_max_abs"] = float(np.max(np.abs(in_domain_values)))
        if name in required_positive and min_value <= 0.0:
            raise ValueError(
                f"OpenFOAM output '{name}' became non-positive during {context}. "
                f"Minimum value: {min_value:.6g}."
            )

    return metrics


def _time_context(raw_value: str) -> str:
    value = raw_value.strip()
    try:
        return f"t={float(value):.12g}"
    except ValueError:
        return f"time={value}"


def _normalise_metric_prefix(label: str) -> str:
    normalised = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    if normalised.endswith("_number"):
        normalised = normalised[: -len("_number")]
    return normalised
