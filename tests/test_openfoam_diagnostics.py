from plm_data.core.health import RuntimeHealthTracker, SolverHealthTracker
from plm_data.presets.fluids._openfoam.diagnostics import (
    ingest_openfoam_log_health,
    sampled_field_runtime_metrics,
)
import numpy as np
import pytest


def test_solver_health_tracker_records_external_observations():
    tracker = SolverHealthTracker(near_iteration_limit_fraction=0.8)

    tracker.record_external_observation(
        name="p",
        context="t=0.1",
        kind="openfoam",
        solver_type="GAMG",
        converged_reason=1,
        iterations=8,
        linear_iterations=8,
        residual_norm=1e-9,
        max_iterations=10,
        initial_residual=0.25,
    )

    report = tracker.build_report()

    assert report["applied"] is True
    assert report["status"] == "warn"
    assert report["near_iteration_limit_solvers"] == ["p"]
    assert report["observed_solvers"] == ["p"]
    summary = report["solvers"]["p"]
    assert summary["solver_type"] == "GAMG"
    assert summary["last_context"] == "t=0.1"
    assert summary["last_iteration_count"] == 8
    assert summary["max_configured_iterations"] == 10
    assert summary["last_initial_residual"] == pytest.approx(0.25)


def test_ingest_openfoam_log_health_populates_shared_trackers():
    log_lines = [
        "Time = 0.005\n",
        "Courant Number mean: 0.12 max: 0.34\n",
        "DILUPBiCGStab:  Solving for Ux, Initial residual = 0.2, Final residual = 1e-07, No Iterations 3\n",
        "DILUPBiCGStab:  Solving for Uy, Initial residual = 0.21, Final residual = 2e-07, No Iterations 3\n",
        "GAMG:  Solving for p, Initial residual = 0.1, Final residual = 3e-06, No Iterations 2\n",
        "time step continuity errors : sum local = 1e-06, global = -2e-18, cumulative = 5e-18\n",
        "Time = 0.01\n",
        "Interface Courant Number mean: 0.02 max: 0.08\n",
        "DILUPBiCGStab:  Solving for T, Initial residual = 0.05, Final residual = 1e-09, No Iterations 1\n",
        "time step continuity errors : sum local = 2e-06, global = 3e-18\n",
    ]
    solver_tracker = SolverHealthTracker()
    runtime_tracker = RuntimeHealthTracker()

    ingest_openfoam_log_health(
        log_lines,
        solver_tracker=solver_tracker,
        runtime_tracker=runtime_tracker,
    )

    solver_report = solver_tracker.build_report()
    runtime_report = runtime_tracker.build_report()

    assert solver_report["applied"] is True
    assert solver_report["status"] == "pass"
    assert solver_report["observed_solvers"] == ["Ux", "Uy", "p", "T"]
    assert solver_report["solvers"]["p"]["last_initial_residual"] == pytest.approx(0.1)

    assert runtime_report["applied"] is True
    assert runtime_report["contexts_checked"] == 2
    assert runtime_report["last_context"] == "t=0.01"
    assert runtime_report["metrics"]["courant_mean"]["last"] == pytest.approx(0.12)
    assert runtime_report["metrics"]["courant_max"]["last"] == pytest.approx(0.34)
    assert runtime_report["metrics"]["interface_courant_mean"]["last"] == pytest.approx(
        0.02
    )
    assert runtime_report["metrics"]["continuity_global_abs"]["max"] == pytest.approx(
        3e-18
    )
    assert "continuity_cumulative_abs" in runtime_report["metrics"]


def test_sampled_field_runtime_metrics_require_positive_fields_and_finite_values():
    mask = np.array([[True, False], [True, True]])
    fields = {
        "density": np.array([[1.0, np.nan], [0.8, 1.2]]),
        "pressure": np.array([[2.0, 5.0], [1.5, 2.5]]),
    }

    metrics = sampled_field_runtime_metrics(
        fields,
        mask,
        positive_fields={"density", "pressure"},
        context="t=0.1",
    )

    assert metrics["sample_valid_fraction"] == pytest.approx(0.75)
    assert metrics["density_min"] == pytest.approx(0.8)
    assert metrics["pressure_max_abs"] == pytest.approx(2.5)

    bad_density = {
        "density": np.array([[1.0, np.nan], [0.0, 1.2]]),
    }
    with pytest.raises(ValueError, match="non-positive"):
        sampled_field_runtime_metrics(
            bad_density,
            mask,
            positive_fields={"density"},
            context="t=0.2",
        )

    bad_temperature = {
        "temperature": np.array([[1.0, np.nan], [np.inf, 1.2]]),
    }
    with pytest.raises(ValueError, match="non-finite"):
        sampled_field_runtime_metrics(
            bad_temperature,
            mask,
            context="t=0.3",
        )
