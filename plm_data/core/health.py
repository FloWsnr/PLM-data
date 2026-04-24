"""Runtime health tracking for solver and PDE-level diagnostics."""

from collections.abc import Mapping
from typing import Any

DEFAULT_SOLVER_NEAR_ITERATION_LIMIT_FRACTION = 0.8

_HEALTH_STATUS_PRIORITY = {
    "pass": 0,
    "warn": 1,
    "fail": 2,
}


def combine_health_status(*statuses: str) -> str:
    """Return the highest-severity status across a collection of checks."""

    if not statuses:
        return "pass"
    return max(statuses, key=lambda status: _HEALTH_STATUS_PRIORITY[status])


def discover_solver_targets(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Discover solver-bearing runtime objects in a problem instance."""

    discovered: dict[str, Any] = {}
    seen: set[int] = set()

    def _visit(value: Any, path: str) -> None:
        value_id = id(value)
        if value_id in seen:
            return

        if isinstance(value, Mapping):
            seen.add(value_id)
            for key, item in value.items():
                _visit(item, f"{path}.{key}")
            return

        if isinstance(value, (list, tuple, set)):
            seen.add(value_id)
            for index, item in enumerate(value):
                _visit(item, f"{path}[{index}]")
            return

        solver = getattr(value, "solver", None)
        if solver is not None and hasattr(solver, "getConvergedReason"):
            seen.add(value_id)
            discovered[path] = value

    for name, value in attributes.items():
        if name in {"_solver_health_tracker", "_runtime_health_tracker"}:
            continue
        _visit(value, name)

    return discovered


class SolverHealthTracker:
    """Aggregate solver convergence and iteration diagnostics across a run."""

    def __init__(
        self,
        *,
        near_iteration_limit_fraction: float = DEFAULT_SOLVER_NEAR_ITERATION_LIMIT_FRACTION,
    ) -> None:
        self._near_iteration_limit_fraction = near_iteration_limit_fraction
        self._solver_summaries: dict[str, dict[str, Any]] = {}

    def record(self, context: str, targets: Mapping[str, Any]) -> None:
        """Observe the active solver state for one or more runtime objects."""

        for name, target in targets.items():
            solver = getattr(target, "solver", target)
            reason = _as_int(_safe_call(solver, "getConvergedReason"))
            if reason is None:
                continue

            iterations = _as_int(_safe_call(solver, "getIterationNumber"))
            linear_iterations = _as_int(_safe_call(solver, "getLinearSolveIterations"))
            residual_norm = _as_float(_safe_call(solver, "getResidualNorm"))
            function_norm = _as_float(_safe_call(solver, "getFunctionNorm"))
            max_iterations = _solver_max_iterations(solver)
            self.record_external_observation(
                name=name,
                context=context,
                kind=_solver_kind(solver),
                solver_type=_safe_call(solver, "getType"),
                converged_reason=reason,
                iterations=iterations,
                linear_iterations=linear_iterations,
                residual_norm=residual_norm,
                function_norm=function_norm,
                max_iterations=max_iterations,
            )

    def record_external_observation(
        self,
        *,
        name: str,
        context: str,
        kind: str,
        solver_type: str | None,
        converged_reason: int,
        iterations: int | None = None,
        linear_iterations: int | None = None,
        residual_norm: float | None = None,
        function_norm: float | None = None,
        max_iterations: int | None = None,
        initial_residual: float | None = None,
    ) -> None:
        """Record one solver observation supplied by a non-PETSc backend."""

        iteration_fraction = (
            float(iterations) / float(max_iterations)
            if iterations is not None and max_iterations not in (None, 0)
            else None
        )
        near_limit = (
            iteration_fraction is not None
            and iteration_fraction >= self._near_iteration_limit_fraction
        )

        summary = self._solver_summaries.setdefault(
            name,
            {
                "kind": kind,
                "solver_type": solver_type,
                "observations": 0,
                "non_converged_observations": 0,
                "near_iteration_limit_observations": 0,
                "last_context": None,
                "last_converged_reason": None,
                "last_iteration_count": None,
                "last_linear_iteration_count": None,
                "last_residual_norm": None,
                "last_function_norm": None,
                "max_iteration_count": None,
                "max_linear_iteration_count": None,
                "max_residual_norm": None,
                "max_function_norm": None,
                "max_iteration_fraction": None,
                "max_configured_iterations": max_iterations,
            },
        )
        if summary["solver_type"] is None and solver_type is not None:
            summary["solver_type"] = solver_type
        if max_iterations is not None:
            summary["max_configured_iterations"] = _max_optional(
                summary["max_configured_iterations"],
                max_iterations,
            )

        summary["observations"] += 1
        if converged_reason <= 0:
            summary["non_converged_observations"] += 1
        if near_limit:
            summary["near_iteration_limit_observations"] += 1

        summary["last_context"] = context
        summary["last_converged_reason"] = converged_reason
        summary["last_iteration_count"] = iterations
        summary["last_linear_iteration_count"] = linear_iterations
        summary["last_residual_norm"] = residual_norm
        summary["last_function_norm"] = function_norm
        summary["max_iteration_count"] = _max_optional(
            summary["max_iteration_count"],
            iterations,
        )
        summary["max_linear_iteration_count"] = _max_optional(
            summary["max_linear_iteration_count"],
            linear_iterations,
        )
        summary["max_residual_norm"] = _max_optional(
            summary["max_residual_norm"],
            residual_norm,
        )
        summary["max_function_norm"] = _max_optional(
            summary["max_function_norm"],
            function_norm,
        )
        summary["max_iteration_fraction"] = _max_optional(
            summary["max_iteration_fraction"],
            iteration_fraction,
        )
        if initial_residual is not None:
            last_initial = _as_float(initial_residual)
            summary["last_initial_residual"] = last_initial
            summary["max_initial_residual"] = _max_optional(
                summary.get("max_initial_residual"),
                last_initial,
            )

    def build_report(self) -> dict[str, Any]:
        """Return a JSON-serializable solver health summary."""

        if not self._solver_summaries:
            return {
                "applied": False,
                "reason": "no_solvers_observed",
                "status": "pass",
                "near_iteration_limit_fraction": self._near_iteration_limit_fraction,
                "observed_solvers": [],
                "near_iteration_limit_solvers": [],
                "non_converged_solvers": [],
                "solvers": {},
            }

        near_limit_solvers = [
            name
            for name, summary in self._solver_summaries.items()
            if summary["near_iteration_limit_observations"] > 0
        ]
        non_converged_solvers = [
            name
            for name, summary in self._solver_summaries.items()
            if summary["non_converged_observations"] > 0
        ]
        status = combine_health_status(
            "fail" if non_converged_solvers else "pass",
            "warn" if near_limit_solvers else "pass",
        )
        return {
            "applied": True,
            "reason": None,
            "status": status,
            "near_iteration_limit_fraction": self._near_iteration_limit_fraction,
            "observed_solvers": list(self._solver_summaries),
            "near_iteration_limit_solvers": near_limit_solvers,
            "non_converged_solvers": non_converged_solvers,
            "solvers": self._solver_summaries,
        }


class RuntimeHealthTracker:
    """Track scalar runtime-health metrics observed during a run."""

    def __init__(self) -> None:
        self._metric_summaries: dict[str, dict[str, Any]] = {}
        self._contexts_checked = 0
        self._last_context: str | None = None

    def record(self, context: str, metrics: Mapping[str, float | int]) -> None:
        """Update aggregate metric summaries for one runtime-health checkpoint."""

        if not metrics:
            return

        self._contexts_checked += 1
        self._last_context = context

        for name, value in metrics.items():
            numeric_value = float(value)
            summary = self._metric_summaries.setdefault(
                name,
                {
                    "num_observations": 0,
                    "min": numeric_value,
                    "max": numeric_value,
                    "last": numeric_value,
                },
            )
            summary["num_observations"] += 1
            summary["min"] = min(summary["min"], numeric_value)
            summary["max"] = max(summary["max"], numeric_value)
            summary["last"] = numeric_value

    def build_report(self) -> dict[str, Any]:
        """Return a JSON-serializable runtime-health summary."""

        if not self._metric_summaries:
            return {
                "applied": False,
                "reason": "no_runtime_checks",
                "status": "pass",
                "contexts_checked": 0,
                "last_context": None,
                "metrics": {},
            }

        return {
            "applied": True,
            "reason": None,
            "status": "pass",
            "contexts_checked": self._contexts_checked,
            "last_context": self._last_context,
            "metrics": self._metric_summaries,
        }


def _solver_kind(solver: Any) -> str:
    """Best-effort solver-kind classification."""

    if hasattr(solver, "getFunctionNorm") or hasattr(
        solver, "getLinearSolveIterations"
    ):
        return "snes"
    return "ksp"


def _solver_max_iterations(solver: Any) -> int | None:
    """Return the configured iteration cap when exposed by the PETSc solver."""

    tolerances = _safe_call(solver, "getTolerances")
    if not isinstance(tolerances, tuple) or len(tolerances) < 4:
        return None
    max_iterations = _as_int(tolerances[3])
    if max_iterations in (None, 0):
        return None
    return max_iterations


def _safe_call(target: Any, method_name: str) -> Any:
    method = getattr(target, method_name, None)
    if method is None:
        return None
    return method()


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _max_optional(
    current: float | int | None, value: float | int | None
) -> float | int | None:
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)
