"""Fast runtime capability coverage for every preset family."""

import numpy as np
import pytest

from plm_data.core.config import BoundaryConditionConfig, TimeConfig
from tests.preset_matrix import (
    RuntimePresetCase,
    assert_expected_output_arrays,
    assert_nontrivial,
    assert_periodic_axis,
    constant,
    make_cahn_hilliard_config,
    make_flow_preset_config,
    make_maxwell_config,
    make_maxwell_pulse_config,
    make_scalar_preset_config,
    run_preset,
    scalar_expr,
    skip_without_complex_runtime,
    skip_without_mpc,
    vector_expr,
    vector_zero,
)


def _mixed_scalar_boundary_conditions() -> dict[str, BoundaryConditionConfig]:
    return {
        "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        "x+": BoundaryConditionConfig(type="neumann", value=constant(0.25)),
        "y-": BoundaryConditionConfig(type="robin", value=constant(0.1), alpha=1.5),
        "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
    }


def _periodic_scalar_boundary_conditions() -> dict[str, BoundaryConditionConfig]:
    return {
        "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
    }


def _lid_velocity_boundary_conditions(
    *, periodic_x: bool = False
) -> dict[str, BoundaryConditionConfig]:
    boundary_conditions = {
        "x-": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
        "x+": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
        "y-": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
        "y+": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(1.0), y=constant(0.0)),
        ),
    }
    if periodic_x:
        del boundary_conditions["x-"]
        del boundary_conditions["x+"]
    return boundary_conditions


def _mixed_maxwell_boundary_conditions() -> dict[str, BoundaryConditionConfig]:
    return {
        "x-": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
        "x+": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
        "y-": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
        "y+": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
    }


def _periodic_maxwell_boundary_conditions() -> dict[str, BoundaryConditionConfig]:
    return {
        "y-": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
        "y+": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
    }


def _maxwell_source():
    return vector_expr(
        x=scalar_expr(
            "gaussian_bump",
            amplitude="param:source_amplitude",
            sigma=0.08,
            center=[0.35, 0.5],
        ),
        y=constant(0.0),
    )


def _maxwell_pulse_source():
    return vector_expr(
        x=scalar_expr("gaussian_bump", amplitude=1.0, sigma=0.06, center=[0.25, 0.5]),
        y=constant(0.0),
    )


def _assert_success(config, result, output_dir):
    assert result.solver_converged is True
    return assert_expected_output_arrays(config, output_dir)


def _assert_scalar_signal(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert_nontrivial(arrays["u"])


def _assert_scalar_periodic_x(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert_nontrivial(arrays["u"])
    assert_periodic_axis(arrays["u"], axis=0)


def _assert_heat_case(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])


def _assert_heat_periodic_x(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_periodic_axis(arrays["u"], axis=0)


def _assert_stokes_source(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert_nontrivial(arrays["velocity_y"])


def _assert_stokes_periodic_x(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert_periodic_axis(arrays["velocity_x"], axis=0)
    assert_periodic_axis(arrays["pressure"], axis=0)


def _assert_navier_source_and_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["velocity_x"])


def _assert_navier_periodic_x(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_periodic_axis(arrays["velocity_x"], axis=0)
    assert_periodic_axis(arrays["velocity_y"], axis=0)
    assert_periodic_axis(arrays["pressure"], axis=0)


def _assert_cahn_constant_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert np.allclose(arrays["c"][0], 0.5, atol=0.05)
    assert_periodic_axis(arrays["c"], axis=0)
    assert_periodic_axis(arrays["c"], axis=1)


def _assert_cahn_random_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert 0.0 < float(np.asarray(arrays["c"]).mean()) < 1.0


def _assert_maxwell_case(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert np.iscomplexobj(arrays["electric_field_x"])
    assert_nontrivial(arrays["electric_field_x"])


def _assert_maxwell_pulse_case(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["electric_field_x"])


SUCCESS_CASES = (
    RuntimePresetCase(
        name="heat_mixed_scalar_bc",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="heat",
            parameters={"kappa": 0.01},
            boundary_conditions=_mixed_scalar_boundary_conditions(),
            source=scalar_expr("sine_product", amplitude=1.0, kx=1, ky=1),
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.1,
                center=[0.5, 0.5],
            ),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_heat_case,
    ),
    RuntimePresetCase(
        name="heat_periodic_x",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="heat",
            parameters={"kappa": 0.01},
            boundary_conditions=_periodic_scalar_boundary_conditions(),
            source=scalar_expr("none"),
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.1,
                center=[0.5, 0.5],
            ),
            periodic_axes=(0,),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_heat_periodic_x,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="poisson_mixed_scalar_bc",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="poisson",
            parameters={"kappa": 1.0, "f_amplitude": 1.0},
            boundary_conditions=_mixed_scalar_boundary_conditions(),
            source=scalar_expr(
                "sine_product",
                amplitude="param:f_amplitude",
                kx=1,
                ky=1,
            ),
        ),
        assert_result=_assert_scalar_signal,
    ),
    RuntimePresetCase(
        name="poisson_periodic_x",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="poisson",
            parameters={"kappa": 1.0, "f_amplitude": 1.0},
            boundary_conditions=_periodic_scalar_boundary_conditions(),
            source=scalar_expr(
                "sine_product",
                amplitude="param:f_amplitude",
                kx=1,
                ky=1,
            ),
            periodic_axes=(0,),
        ),
        assert_result=_assert_scalar_periodic_x,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="helmholtz_mixed_scalar_bc",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="helmholtz",
            parameters={"kappa": 1.0, "k": 2.0, "f_amplitude": 1.0},
            boundary_conditions=_mixed_scalar_boundary_conditions(),
            source=scalar_expr(
                "sine_product",
                amplitude="param:f_amplitude",
                kx=1,
                ky=1,
            ),
        ),
        assert_result=_assert_scalar_signal,
    ),
    RuntimePresetCase(
        name="helmholtz_periodic_x",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="helmholtz",
            parameters={"kappa": 1.0, "k": 2.0, "f_amplitude": 1.0},
            boundary_conditions=_periodic_scalar_boundary_conditions(),
            source=scalar_expr(
                "sine_product",
                amplitude="param:f_amplitude",
                kx=1,
                ky=1,
            ),
            periodic_axes=(0,),
        ),
        assert_result=_assert_scalar_periodic_x,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="stokes_body_force",
        make_config=lambda tmp_path: make_flow_preset_config(
            tmp_path,
            preset="stokes",
            parameters={"nu": 1.0},
            boundary_conditions=_lid_velocity_boundary_conditions(),
            source=vector_expr(x=constant(0.0), y=constant(-1.0)),
        ),
        assert_result=_assert_stokes_source,
    ),
    RuntimePresetCase(
        name="stokes_periodic_x",
        make_config=lambda tmp_path: make_flow_preset_config(
            tmp_path,
            preset="stokes",
            parameters={"nu": 1.0},
            boundary_conditions=_lid_velocity_boundary_conditions(periodic_x=True),
            source=scalar_expr("none"),
            periodic_axes=(0,),
        ),
        assert_result=_assert_stokes_periodic_x,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="navier_stokes_source_and_vector_ic",
        make_config=lambda tmp_path: make_flow_preset_config(
            tmp_path,
            preset="navier_stokes",
            parameters={"Re": 25.0, "k": 1.0},
            boundary_conditions=_lid_velocity_boundary_conditions(),
            source=vector_expr(x=constant(0.0), y=constant(-1.0)),
            initial_condition=vector_expr(
                x=scalar_expr(
                    "gaussian_bump",
                    amplitude=0.5,
                    sigma=0.2,
                    center=[0.5, 0.5],
                ),
                y=constant(0.0),
            ),
            time=TimeConfig(dt=0.1, t_end=0.1),
        ),
        assert_result=_assert_navier_source_and_ic,
    ),
    RuntimePresetCase(
        name="navier_stokes_periodic_x_custom_ic",
        make_config=lambda tmp_path: make_flow_preset_config(
            tmp_path,
            preset="navier_stokes",
            parameters={"Re": 25.0, "k": 1.0},
            boundary_conditions=_lid_velocity_boundary_conditions(periodic_x=True),
            source=scalar_expr("none"),
            initial_condition=scalar_expr("custom"),
            periodic_axes=(0,),
            time=TimeConfig(dt=0.1, t_end=0.1),
        ),
        assert_result=_assert_navier_periodic_x,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="cahn_hilliard_constant_ic",
        make_config=lambda tmp_path: make_cahn_hilliard_config(
            tmp_path,
            initial_condition=constant(0.5),
        ),
        assert_result=_assert_cahn_constant_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="cahn_hilliard_random_perturbation_ic",
        make_config=lambda tmp_path: make_cahn_hilliard_config(
            tmp_path,
            initial_condition=scalar_expr(
                "random_perturbation",
                mean=0.63,
                std=0.02,
            ),
        ),
        assert_result=_assert_cahn_random_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="maxwell_mixed_boundaries",
        make_config=lambda tmp_path: make_maxwell_config(
            tmp_path,
            boundary_conditions=_mixed_maxwell_boundary_conditions(),
            source=_maxwell_source(),
        ),
        assert_result=_assert_maxwell_case,
        skip_reason=skip_without_complex_runtime,
    ),
    RuntimePresetCase(
        name="maxwell_pulse_mixed_boundaries",
        make_config=lambda tmp_path: make_maxwell_pulse_config(
            tmp_path,
            boundary_conditions=_mixed_maxwell_boundary_conditions(),
            source=_maxwell_pulse_source(),
            initial_condition=vector_zero(),
        ),
        assert_result=_assert_maxwell_pulse_case,
    ),
)


REJECTION_CASES = (
    RuntimePresetCase(
        name="maxwell_periodic_domain_rejected",
        make_config=lambda tmp_path: make_maxwell_config(
            tmp_path,
            boundary_conditions=_periodic_maxwell_boundary_conditions(),
            source=_maxwell_source(),
            periodic_axes=(0,),
        ),
        expected_error=NotImplementedError,
        expected_error_match="N1curl spaces",
        skip_reason=skip_without_complex_runtime,
    ),
    RuntimePresetCase(
        name="maxwell_pulse_periodic_domain_rejected",
        make_config=lambda tmp_path: make_maxwell_pulse_config(
            tmp_path,
            boundary_conditions=_periodic_maxwell_boundary_conditions(),
            source=_maxwell_pulse_source(),
            initial_condition=vector_zero(),
            periodic_axes=(0,),
        ),
        expected_error=NotImplementedError,
        expected_error_match="N1curl spaces",
    ),
)


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=lambda case: case.name)
def test_preset_capability_matrix_success(case, tmp_path):
    reason = case.skip_reason()
    if reason is not None:
        pytest.skip(reason)

    config = case.make_config(tmp_path)
    result, output_dir = run_preset(config)

    assert case.assert_result is not None
    case.assert_result(config, result, output_dir)


@pytest.mark.parametrize("case", REJECTION_CASES, ids=lambda case: case.name)
def test_preset_capability_matrix_rejections(case, tmp_path):
    reason = case.skip_reason()
    if reason is not None:
        pytest.skip(reason)

    config = case.make_config(tmp_path)

    assert case.expected_error is not None
    with pytest.raises(case.expected_error, match=case.expected_error_match):
        run_preset(config)
