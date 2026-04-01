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
    make_advection_config,
    make_burgers_config,
    make_cyclic_competition_config,
    make_gierer_meinhardt_config,
    make_immunotherapy_config,
    make_cahn_hilliard_config,
    make_cgl_config,
    make_kuramoto_sivashinsky_config,
    make_zakharov_kuznetsov_config,
    make_flow_preset_config,
    make_gray_scott_config,
    make_lorenz_config,
    make_maxwell_config,
    make_maxwell_pulse_config,
    make_plate_config,
    make_shallow_water_config,
    make_scalar_preset_config,
    make_swift_hohenberg_config,
    make_superlattice_config,
    make_thermal_convection_config,
    make_van_der_pol_config,
    make_wave_config,
    run_preset,
    scalar_expr,
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


def _simply_supported_plate_boundary_conditions(
    *, gdim: int
) -> dict[str, BoundaryConditionConfig]:
    if gdim == 2:
        sides = ("x-", "x+", "y-", "y+")
    elif gdim == 3:
        sides = ("x-", "x+", "y-", "y+", "z-", "z+")
    else:
        raise ValueError(f"Plate boundary helper only supports 2D/3D, got {gdim}D")
    return {side: BoundaryConditionConfig(type="simply_supported") for side in sides}


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


def _mixed_burgers_boundary_conditions() -> dict[str, BoundaryConditionConfig]:
    return {
        "x-": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
        "x+": BoundaryConditionConfig(
            type="neumann",
            value=vector_expr(x=constant(0.15), y=constant(-0.05)),
        ),
        "y-": BoundaryConditionConfig(
            type="neumann",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
        "y+": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
    }


def _burgers_periodic_ic(*, gdim: int):
    if gdim == 2:
        return vector_expr(
            x=scalar_expr("sine_product", amplitude=1.0, ky=2.0),
            y=scalar_expr("sine_product", amplitude=-1.0, kx=2.0),
        )
    return vector_expr(
        x=scalar_expr("sine_product", amplitude=0.9, ky=2.0),
        y=scalar_expr("sine_product", amplitude=0.9, kz=2.0),
        z=scalar_expr("sine_product", amplitude=-0.9, kx=2.0),
    )


def _thermal_velocity_boundary_conditions(
    *, gdim: int, periodic_axes: tuple[int, ...]
) -> dict[str, BoundaryConditionConfig]:
    if gdim == 2:
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
                value=vector_expr(x=constant(0.0), y=constant(0.0)),
            ),
        }
    else:
        boundary_conditions = {
            "x-": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
            "x+": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
            "y-": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
            "y+": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
            "z-": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
            "z+": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
        }

    for axis in periodic_axes:
        minus, plus = {0: ("x-", "x+"), 1: ("y-", "y+"), 2: ("z-", "z+")}[axis]
        boundary_conditions.pop(minus, None)
        boundary_conditions.pop(plus, None)

    return boundary_conditions


def _thermal_temperature_boundary_conditions(
    *, gdim: int, periodic_axes: tuple[int, ...]
) -> dict[str, BoundaryConditionConfig]:
    if gdim == 2:
        boundary_conditions = {
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="dirichlet", value=constant(1.0)),
            "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        }
    else:
        boundary_conditions = {
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "z-": BoundaryConditionConfig(type="dirichlet", value=constant(1.0)),
            "z+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        }

    for axis in periodic_axes:
        minus, plus = {0: ("x-", "x+"), 1: ("y-", "y+"), 2: ("z-", "z+")}[axis]
        boundary_conditions.pop(minus, None)
        boundary_conditions.pop(plus, None)

    return boundary_conditions


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


def _assert_advection_periodic_xy(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)


def _assert_advection_periodic_xyz(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)
    assert_periodic_axis(arrays["u"], axis=2)


def _assert_plate_case(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["deflection"])
    assert_nontrivial(arrays["velocity"])


def _assert_wave_case(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_nontrivial(arrays["v"])


def _assert_wave_periodic_x(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_nontrivial(arrays["v"])
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["v"], axis=0)


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


def _assert_burgers_mixed_bc(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["velocity_x"])
    assert_nontrivial(arrays["velocity_y"])


def _assert_burgers_periodic_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("velocity_x", "velocity_y"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)


def _assert_burgers_periodic_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("velocity_x", "velocity_y", "velocity_z"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)
        assert_periodic_axis(arrays[field_name], axis=2)


def _assert_cgl_constant_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert np.allclose(arrays["u"][0], 1.0, atol=0.1)
    assert np.allclose(arrays["v"][0], 0.0, atol=0.1)
    assert np.all(np.isfinite(arrays["amplitude"]))
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)


def _assert_cgl_random_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert np.all(np.isfinite(arrays["u"]))
    assert np.all(np.isfinite(arrays["v"]))
    assert np.all(np.isfinite(arrays["amplitude"]))
    assert np.all(arrays["amplitude"] >= 0)


def _assert_ks_constant_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert np.allclose(arrays["u"][0], 0.0, atol=0.05)
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)


def _assert_ks_random_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert np.all(np.isfinite(arrays["u"]))


def _assert_zk_constant_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert np.allclose(arrays["u"][0], 0.0, atol=0.05)
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)


def _assert_zk_random_ic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert np.all(np.isfinite(arrays["u"]))


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


def _assert_gray_scott_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_nontrivial(arrays["v"])
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)
    assert_periodic_axis(arrays["v"], axis=0)
    assert_periodic_axis(arrays["v"], axis=1)


def _assert_gray_scott_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_nontrivial(arrays["v"])
    for field_name in ("u", "v"):
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)
        assert_periodic_axis(arrays[field_name], axis=2)


def _assert_swift_hohenberg_periodic_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)


def _assert_swift_hohenberg_periodic_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)
    assert_periodic_axis(arrays["u"], axis=2)


def _assert_swift_hohenberg_wall_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    edge_max = max(
        float(np.max(np.abs(arrays["u"][-1, 0, :]))),
        float(np.max(np.abs(arrays["u"][-1, -1, :]))),
        float(np.max(np.abs(arrays["u"][-1, :, 0]))),
        float(np.max(np.abs(arrays["u"][-1, :, -1]))),
    )
    assert edge_max < 5.0e-2


def _assert_lorenz_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("x", "y", "z"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)


def _assert_lorenz_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("x", "y", "z"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)
        assert_periodic_axis(arrays[field_name], axis=2)


def _assert_gierer_meinhardt_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("a", "h"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0, frame=0)
        assert_periodic_axis(arrays[field_name], axis=1, frame=0)
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)


def _assert_gierer_meinhardt_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("a", "h"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0, frame=0)
        assert_periodic_axis(arrays[field_name], axis=1, frame=0)
        assert_periodic_axis(arrays[field_name], axis=2, frame=0)
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)
        assert_periodic_axis(arrays[field_name], axis=2)


def _assert_gierer_meinhardt_mixed_bc(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("a", "h"):
        assert_nontrivial(arrays[field_name])
    # h has Dirichlet zero on all sides
    assert np.allclose(arrays["h"][-1, 0, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["h"][-1, -1, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["h"][-1, :, 0], 0.0, atol=5e-3)
    assert np.allclose(arrays["h"][-1, :, -1], 0.0, atol=5e-3)


def _assert_cyclic_competition_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v", "w"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0, frame=0)
        assert_periodic_axis(arrays[field_name], axis=1, frame=0)
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)


def _assert_cyclic_competition_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v", "w"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0, frame=0)
        assert_periodic_axis(arrays[field_name], axis=1, frame=0)
        assert_periodic_axis(arrays[field_name], axis=2, frame=0)
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)
        assert_periodic_axis(arrays[field_name], axis=2)


def _assert_cyclic_competition_mixed_bc(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v", "w"):
        assert_nontrivial(arrays[field_name])
    # w has Dirichlet zero on all sides
    assert np.allclose(arrays["w"][-1, 0, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["w"][-1, -1, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["w"][-1, :, 0], 0.0, atol=5e-3)
    assert np.allclose(arrays["w"][-1, :, -1], 0.0, atol=5e-3)


def _assert_immunotherapy_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v", "w"):
        assert_nontrivial(arrays[field_name])


def _assert_immunotherapy_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v", "w"):
        assert_nontrivial(arrays[field_name])


def _assert_immunotherapy_periodic(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v", "w"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0, frame=0)
        assert_periodic_axis(arrays[field_name], axis=1, frame=0)
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)


def _assert_immunotherapy_mixed_bc(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v", "w"):
        assert_nontrivial(arrays[field_name])
    # w has Dirichlet zero on all sides
    assert np.allclose(arrays["w"][-1, 0, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["w"][-1, -1, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["w"][-1, :, 0], 0.0, atol=5e-3)
    assert np.allclose(arrays["w"][-1, :, -1], 0.0, atol=5e-3)


def _assert_van_der_pol_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_nontrivial(arrays["v"])
    assert_periodic_axis(arrays["u"], axis=0, frame=0)
    assert_periodic_axis(arrays["u"], axis=1, frame=0)
    assert_periodic_axis(arrays["v"], axis=0, frame=0)
    assert_periodic_axis(arrays["v"], axis=1, frame=0)
    assert_periodic_axis(arrays["u"], axis=0)
    assert_periodic_axis(arrays["u"], axis=1)
    assert_periodic_axis(arrays["v"], axis=0)
    assert_periodic_axis(arrays["v"], axis=1)


def _assert_van_der_pol_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u", "v"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0, frame=0)
        assert_periodic_axis(arrays[field_name], axis=1, frame=0)
        assert_periodic_axis(arrays[field_name], axis=2, frame=0)
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)
        assert_periodic_axis(arrays[field_name], axis=2)


def _assert_van_der_pol_mixed_bc(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"])
    assert_nontrivial(arrays["v"])
    assert np.allclose(arrays["v"][-1, 0, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["v"][-1, -1, :], 0.0, atol=5e-3)
    assert np.allclose(arrays["v"][-1, :, 0], 0.0, atol=5e-3)
    assert np.allclose(arrays["v"][-1, :, -1], 0.0, atol=5e-3)


def _assert_superlattice_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u_1", "v_1", "u_2", "v_2"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)


def _assert_superlattice_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u_1", "v_1", "u_2", "v_2"):
        assert_nontrivial(arrays[field_name])
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)
        assert_periodic_axis(arrays[field_name], axis=2)


def _assert_superlattice_mixed_bc(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    for field_name in ("u_1", "v_1", "u_2", "v_2"):
        assert_nontrivial(arrays[field_name])


def _assert_maxwell_case(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert not np.iscomplexobj(arrays["electric_field_real_x"])
    assert not np.iscomplexobj(arrays["electric_field_imag_x"])
    assert_nontrivial(arrays["electric_field_real_x"])
    assert_nontrivial(arrays["electric_field_imag_x"])


def _assert_maxwell_pulse_case(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["electric_field_x"])


def _assert_thermal_convection_2d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["temperature"])
    assert_periodic_axis(arrays["velocity_x"], axis=0)
    assert_periodic_axis(arrays["velocity_y"], axis=0)
    assert_periodic_axis(arrays["pressure"], axis=0)
    assert_periodic_axis(arrays["temperature"], axis=0)


def _assert_thermal_convection_3d(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["temperature"])
    for field_name in (
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "pressure",
        "temperature",
    ):
        assert_periodic_axis(arrays[field_name], axis=0)
        assert_periodic_axis(arrays[field_name], axis=1)


def _assert_shallow_water_periodic_xy(config, result, output_dir):
    arrays = _assert_success(config, result, output_dir)
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["height"])
    assert_nontrivial(arrays["velocity_x"])
    assert_periodic_axis(arrays["height"], axis=0)
    assert_periodic_axis(arrays["height"], axis=1)
    assert_periodic_axis(arrays["velocity_x"], axis=0)
    assert_periodic_axis(arrays["velocity_x"], axis=1)
    assert_periodic_axis(arrays["velocity_y"], axis=0)
    assert_periodic_axis(arrays["velocity_y"], axis=1)


SUCCESS_CASES = (
    RuntimePresetCase(
        name="wave_mixed_scalar_bc",
        make_config=lambda tmp_path: make_wave_config(
            tmp_path,
            gdim=2,
            boundary_conditions=_mixed_scalar_boundary_conditions(),
            initial_displacement=constant(0.0),
            initial_velocity=constant(0.0),
            forcing=scalar_expr("sine_product", amplitude=1.0, kx=1, ky=1),
            time=TimeConfig(dt=0.02, t_end=0.02),
        ),
        assert_result=_assert_wave_case,
    ),
    RuntimePresetCase(
        name="wave_periodic_x",
        make_config=lambda tmp_path: make_wave_config(
            tmp_path,
            gdim=2,
            boundary_conditions=_periodic_scalar_boundary_conditions(),
            initial_displacement=scalar_expr(
                "sine_product",
                amplitude=1.0,
                kx=2,
                ky=1,
            ),
            initial_velocity=constant(0.0),
            forcing=scalar_expr("none"),
            periodic_axes=(0,),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_wave_periodic_x,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="heat_mixed_scalar_bc",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="heat",
            parameters={},
            coefficients={"kappa": constant(0.01)},
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
        name="plate_simply_supported",
        make_config=lambda tmp_path: make_plate_config(
            tmp_path,
            gdim=2,
            boundary_conditions=_simply_supported_plate_boundary_conditions(gdim=2),
            deflection_initial_condition=constant(0.0),
            velocity_initial_condition=constant(0.0),
            load_source=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.15,
                center=[0.5, 0.5],
            ),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        assert_result=_assert_plate_case,
    ),
    RuntimePresetCase(
        name="heat_periodic_x",
        make_config=lambda tmp_path: make_scalar_preset_config(
            tmp_path,
            preset="heat",
            parameters={},
            coefficients={"kappa": constant(0.01)},
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
        name="advection_periodic_xy_pure",
        make_config=lambda tmp_path: make_advection_config(
            tmp_path,
            gdim=2,
            velocity=vector_expr(
                x=constant(1.0),
                y=constant(0.35),
            ),
            diffusivity=scalar_expr("zero"),
            boundary_conditions={},
            source=scalar_expr("none"),
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.08,
                center=[0.24, 0.39],
            ),
            periodic_axes=(0, 1),
            mesh_resolution=(14, 14),
            output_resolution=(6, 6),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_advection_periodic_xy,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="advection_periodic_xy_diffusive",
        make_config=lambda tmp_path: make_advection_config(
            tmp_path,
            gdim=2,
            velocity=vector_expr(
                x=scalar_expr("sine_product", amplitude=1.0, ky=2.0),
                y=scalar_expr("sine_product", amplitude=-1.0, kx=2.0),
            ),
            diffusivity=constant(0.001),
            boundary_conditions={},
            source=scalar_expr("none"),
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.08,
                center=[0.31, 0.57],
            ),
            periodic_axes=(0, 1),
            mesh_resolution=(14, 14),
            output_resolution=(6, 6),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_advection_periodic_xy,
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
        name="burgers_mixed_vector_bc",
        make_config=lambda tmp_path: make_burgers_config(
            tmp_path,
            gdim=2,
            parameters={"nu": 0.02},
            boundary_conditions=_mixed_burgers_boundary_conditions(),
            source=scalar_expr("none"),
            initial_condition=vector_expr(
                x=scalar_expr(
                    "gaussian_bump",
                    amplitude=0.8,
                    sigma=0.12,
                    center=[0.35, 0.45],
                ),
                y=scalar_expr("sine_product", amplitude=0.2, kx=1.0, ky=1.0),
            ),
            mesh_resolution=(12, 12),
            output_resolution=(6, 6),
            time=TimeConfig(dt=0.02, t_end=0.02),
        ),
        assert_result=_assert_burgers_mixed_bc,
    ),
    RuntimePresetCase(
        name="burgers_2d_periodic_xy",
        make_config=lambda tmp_path: make_burgers_config(
            tmp_path,
            gdim=2,
            parameters={"nu": 0.005},
            boundary_conditions={},
            source=scalar_expr("none"),
            initial_condition=_burgers_periodic_ic(gdim=2),
            periodic_axes=(0, 1),
            mesh_resolution=(12, 12),
            output_resolution=(6, 6),
            time=TimeConfig(dt=0.02, t_end=0.02),
        ),
        assert_result=_assert_burgers_periodic_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="burgers_3d_periodic_xyz",
        make_config=lambda tmp_path: make_burgers_config(
            tmp_path,
            gdim=3,
            parameters={"nu": 0.01},
            boundary_conditions={},
            source=scalar_expr("none"),
            initial_condition=_burgers_periodic_ic(gdim=3),
            periodic_axes=(0, 1, 2),
            mesh_resolution=(6, 6, 6),
            output_resolution=(4, 4, 4),
            time=TimeConfig(dt=0.02, t_end=0.02),
        ),
        assert_result=_assert_burgers_periodic_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="thermal_convection_2d_periodic_x",
        make_config=lambda tmp_path: make_thermal_convection_config(
            tmp_path,
            gdim=2,
            parameters={"Ra": 1000.0, "Pr": 1.0, "k": 1.0},
            velocity_boundary_conditions=_thermal_velocity_boundary_conditions(
                gdim=2,
                periodic_axes=(0,),
            ),
            temperature_boundary_conditions=_thermal_temperature_boundary_conditions(
                gdim=2,
                periodic_axes=(0,),
            ),
            velocity_source=scalar_expr("none"),
            velocity_initial_condition=vector_zero(),
            temperature_source=scalar_expr("none"),
            temperature_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=0.5,
                std=0.02,
            ),
            periodic_axes=(0,),
            mesh_resolution=(6, 6),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        assert_result=_assert_thermal_convection_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="thermal_convection_3d_periodic_xy",
        make_config=lambda tmp_path: make_thermal_convection_config(
            tmp_path,
            gdim=3,
            parameters={"Ra": 1000.0, "Pr": 1.0, "k": 1.0},
            velocity_boundary_conditions=_thermal_velocity_boundary_conditions(
                gdim=3,
                periodic_axes=(0, 1),
            ),
            temperature_boundary_conditions=_thermal_temperature_boundary_conditions(
                gdim=3,
                periodic_axes=(0, 1),
            ),
            velocity_source=scalar_expr("none"),
            velocity_initial_condition=vector_zero(),
            temperature_source=scalar_expr("none"),
            temperature_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=0.5,
                std=0.02,
            ),
            periodic_axes=(0, 1),
            mesh_resolution=(4, 4, 3),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        assert_result=_assert_thermal_convection_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="shallow_water_periodic_xy",
        make_config=lambda tmp_path: make_shallow_water_config(
            tmp_path,
            parameters={
                "gravity": 1.0,
                "mean_depth": 1.0,
                "drag": 0.01,
                "viscosity": 0.002,
                "coriolis": 0.2,
            },
            bathymetry=constant(0.0),
            initial_height=scalar_expr(
                "gaussian_bump",
                amplitude=0.08,
                sigma=0.1,
                center=[0.35, 0.45],
            ),
            initial_velocity=vector_zero(),
            height_boundary_conditions={},
            velocity_boundary_conditions={},
            height_periodic_axes=(0, 1),
            velocity_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(5, 5),
            time=TimeConfig(dt=0.02, t_end=0.02),
        ),
        assert_result=_assert_shallow_water_periodic_xy,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="advection_3d_periodic_xyz",
        make_config=lambda tmp_path: make_advection_config(
            tmp_path,
            gdim=3,
            velocity=vector_expr(
                x=scalar_expr("sine_product", amplitude=0.9, ky=2.0),
                y=scalar_expr("sine_product", amplitude=0.9, kz=2.0),
                z=scalar_expr("sine_product", amplitude=0.9, kx=2.0),
            ),
            diffusivity=constant(0.001),
            boundary_conditions={},
            source=scalar_expr("none"),
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.1,
                center=[0.3, 0.45, 0.6],
            ),
            periodic_axes=(0, 1, 2),
            mesh_resolution=(8, 8, 8),
            output_resolution=(4, 4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_advection_periodic_xyz,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="cgl_constant_ic",
        make_config=lambda tmp_path: make_cgl_config(
            tmp_path,
            u_initial_condition=constant(1.0),
            v_initial_condition=constant(0.0),
        ),
        assert_result=_assert_cgl_constant_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="cgl_gaussian_noise_ic",
        make_config=lambda tmp_path: make_cgl_config(
            tmp_path,
            u_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=1.0,
                std=0.05,
            ),
            v_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=0.0,
                std=0.05,
            ),
        ),
        assert_result=_assert_cgl_random_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="kuramoto_sivashinsky_constant_ic",
        make_config=lambda tmp_path: make_kuramoto_sivashinsky_config(
            tmp_path,
            initial_condition=constant(0.0),
        ),
        assert_result=_assert_ks_constant_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="kuramoto_sivashinsky_gaussian_noise_ic",
        make_config=lambda tmp_path: make_kuramoto_sivashinsky_config(
            tmp_path,
            initial_condition=scalar_expr(
                "gaussian_noise",
                mean=0.0,
                std=0.01,
            ),
        ),
        assert_result=_assert_ks_random_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="zakharov_kuznetsov_constant_ic",
        make_config=lambda tmp_path: make_zakharov_kuznetsov_config(
            tmp_path,
            initial_condition=constant(0.0),
        ),
        assert_result=_assert_zk_constant_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="zakharov_kuznetsov_gaussian_noise_ic",
        make_config=lambda tmp_path: make_zakharov_kuznetsov_config(
            tmp_path,
            initial_condition=scalar_expr(
                "gaussian_noise",
                mean=0.0,
                std=0.01,
            ),
        ),
        assert_result=_assert_zk_random_ic,
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
        name="cahn_hilliard_gaussian_noise_ic",
        make_config=lambda tmp_path: make_cahn_hilliard_config(
            tmp_path,
            initial_condition=scalar_expr(
                "gaussian_noise",
                mean=0.63,
                std=0.02,
            ),
        ),
        assert_result=_assert_cahn_random_ic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="gray_scott_2d_periodic_xy",
        make_config=lambda tmp_path: make_gray_scott_config(
            tmp_path,
            gdim=2,
            velocity=vector_expr(x=constant(0.35), y=constant(0.0)),
            u_initial_condition=scalar_expr(
                "gaussian_blobs",
                background=1.0,
                blobs=[{"amplitude": -0.5, "sigma": 0.08, "center": [0.5, 0.5]}],
            ),
            v_initial_condition=scalar_expr(
                "gaussian_blobs",
                background=0.0,
                blobs=[{"amplitude": 0.25, "sigma": 0.08, "center": [0.5, 0.5]}],
            ),
            u_boundary_conditions={},
            v_boundary_conditions={},
            u_periodic_axes=(0, 1),
            v_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=1.0, t_end=1.0),
        ),
        assert_result=_assert_gray_scott_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="gray_scott_3d_periodic_xyz",
        make_config=lambda tmp_path: make_gray_scott_config(
            tmp_path,
            gdim=3,
            velocity=vector_expr(
                x=constant(0.2),
                y=constant(0.0),
                z=constant(0.1),
            ),
            u_initial_condition=scalar_expr(
                "gaussian_blobs",
                background=1.0,
                blobs=[
                    {
                        "amplitude": -0.5,
                        "sigma": 0.08,
                        "center": [0.5, 0.5, 0.5],
                    }
                ],
            ),
            v_initial_condition=scalar_expr(
                "gaussian_blobs",
                background=0.0,
                blobs=[
                    {
                        "amplitude": 0.25,
                        "sigma": 0.08,
                        "center": [0.5, 0.5, 0.5],
                    }
                ],
            ),
            u_boundary_conditions={},
            v_boundary_conditions={},
            u_periodic_axes=(0, 1, 2),
            v_periodic_axes=(0, 1, 2),
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=1.0, t_end=1.0),
        ),
        assert_result=_assert_gray_scott_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="swift_hohenberg_2d_periodic_xy",
        make_config=lambda tmp_path: make_swift_hohenberg_config(
            tmp_path,
            gdim=2,
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.12,
                center=[0.5, 0.5],
            ),
            velocity=vector_expr(x=constant(0.6), y=constant(-0.25)),
            boundary_conditions={},
            periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        assert_result=_assert_swift_hohenberg_periodic_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="swift_hohenberg_2d_simply_supported",
        make_config=lambda tmp_path: make_swift_hohenberg_config(
            tmp_path,
            gdim=2,
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.12,
                center=[0.5, 0.5],
            ),
            velocity=vector_expr(
                x=scalar_expr("affine", constant=0.15, y=-0.3),
                y=scalar_expr("affine", constant=-0.15, x=0.3),
            ),
            boundary_conditions=_simply_supported_plate_boundary_conditions(gdim=2),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        assert_result=_assert_swift_hohenberg_wall_2d,
    ),
    RuntimePresetCase(
        name="swift_hohenberg_3d_periodic_xyz",
        make_config=lambda tmp_path: make_swift_hohenberg_config(
            tmp_path,
            gdim=3,
            initial_condition=scalar_expr(
                "gaussian_bump",
                amplitude=1.0,
                sigma=0.18,
                center=[0.5, 0.5, 0.5],
            ),
            velocity=vector_expr(
                x=constant(0.25),
                y=constant(-0.1),
                z=constant(0.15),
            ),
            boundary_conditions={},
            periodic_axes=(0, 1, 2),
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        assert_result=_assert_swift_hohenberg_periodic_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="lorenz_2d_periodic_xy",
        make_config=lambda tmp_path: make_lorenz_config(
            tmp_path,
            gdim=2,
            x_initial_condition=scalar_expr("gaussian_noise", mean=0.0, std=1.0),
            y_initial_condition=scalar_expr("gaussian_noise", mean=0.0, std=1.0),
            z_initial_condition=scalar_expr("gaussian_noise", mean=0.0, std=1.0),
            x_boundary_conditions={},
            y_boundary_conditions={},
            z_boundary_conditions={},
            x_periodic_axes=(0, 1),
            y_periodic_axes=(0, 1),
            z_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.001, t_end=0.001),
        ),
        assert_result=_assert_lorenz_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="lorenz_3d_periodic_xyz",
        make_config=lambda tmp_path: make_lorenz_config(
            tmp_path,
            gdim=3,
            x_initial_condition=scalar_expr("gaussian_noise", mean=0.0, std=1.0),
            y_initial_condition=scalar_expr("gaussian_noise", mean=0.0, std=1.0),
            z_initial_condition=scalar_expr("gaussian_noise", mean=0.0, std=1.0),
            x_boundary_conditions={},
            y_boundary_conditions={},
            z_boundary_conditions={},
            x_periodic_axes=(0, 1, 2),
            y_periodic_axes=(0, 1, 2),
            z_periodic_axes=(0, 1, 2),
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.001, t_end=0.001),
        ),
        assert_result=_assert_lorenz_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="gierer_meinhardt_2d_periodic_xy",
        make_config=lambda tmp_path: make_gierer_meinhardt_config(
            tmp_path,
            gdim=2,
            a_initial_condition=scalar_expr("gaussian_noise", mean=1.0, std=0.05),
            h_initial_condition=scalar_expr("gaussian_noise", mean=1.0, std=0.05),
            a_boundary_conditions={},
            h_boundary_conditions={},
            a_periodic_axes=(0, 1),
            h_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_gierer_meinhardt_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="gierer_meinhardt_3d_periodic_xyz",
        make_config=lambda tmp_path: make_gierer_meinhardt_config(
            tmp_path,
            gdim=3,
            a_initial_condition=scalar_expr("gaussian_noise", mean=1.0, std=0.05),
            h_initial_condition=scalar_expr("gaussian_noise", mean=1.0, std=0.05),
            a_boundary_conditions={},
            h_boundary_conditions={},
            a_periodic_axes=(0, 1, 2),
            h_periodic_axes=(0, 1, 2),
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_gierer_meinhardt_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="gierer_meinhardt_2d_field_specific_scalar_boundaries",
        make_config=lambda tmp_path: make_gierer_meinhardt_config(
            tmp_path,
            gdim=2,
            a_initial_condition=scalar_expr("gaussian_noise", mean=1.0, std=0.05),
            h_initial_condition=scalar_expr("gaussian_noise", mean=1.0, std=0.05),
            a_boundary_conditions={},
            h_boundary_conditions={
                "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            },
            a_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_gierer_meinhardt_mixed_bc,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="cyclic_competition_2d_periodic_xy",
        make_config=lambda tmp_path: make_cyclic_competition_config(
            tmp_path,
            gdim=2,
            u_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            v_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            w_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            u_boundary_conditions={},
            v_boundary_conditions={},
            w_boundary_conditions={},
            u_periodic_axes=(0, 1),
            v_periodic_axes=(0, 1),
            w_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_cyclic_competition_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="cyclic_competition_3d_periodic_xyz",
        make_config=lambda tmp_path: make_cyclic_competition_config(
            tmp_path,
            gdim=3,
            u_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            v_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            w_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            u_boundary_conditions={},
            v_boundary_conditions={},
            w_boundary_conditions={},
            u_periodic_axes=(0, 1, 2),
            v_periodic_axes=(0, 1, 2),
            w_periodic_axes=(0, 1, 2),
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_cyclic_competition_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="cyclic_competition_2d_field_specific_scalar_boundaries",
        make_config=lambda tmp_path: make_cyclic_competition_config(
            tmp_path,
            gdim=2,
            u_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            v_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            w_initial_condition=scalar_expr("gaussian_noise", mean=0.385, std=0.05),
            u_boundary_conditions={},
            v_boundary_conditions={},
            w_boundary_conditions={
                "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            },
            u_periodic_axes=(0, 1),
            v_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_cyclic_competition_mixed_bc,
        skip_reason=skip_without_mpc,
    ),
    # --- immunotherapy ---
    RuntimePresetCase(
        name="immunotherapy_2d_neumann",
        make_config=lambda tmp_path: make_immunotherapy_config(
            tmp_path,
            gdim=2,
            u_initial_condition=scalar_expr("gaussian_noise", mean=0.3, std=0.05),
            v_initial_condition=scalar_expr("gaussian_noise", mean=0.9, std=0.05),
            w_initial_condition=scalar_expr("gaussian_noise", mean=0.01, std=0.002),
            u_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            v_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            w_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.1, t_end=0.1),
        ),
        assert_result=_assert_immunotherapy_2d,
    ),
    RuntimePresetCase(
        name="immunotherapy_3d_neumann",
        make_config=lambda tmp_path: make_immunotherapy_config(
            tmp_path,
            gdim=3,
            u_initial_condition=scalar_expr("gaussian_noise", mean=0.3, std=0.05),
            v_initial_condition=scalar_expr("gaussian_noise", mean=0.9, std=0.05),
            w_initial_condition=scalar_expr("gaussian_noise", mean=0.01, std=0.002),
            u_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "z-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "z+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            v_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "z-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "z+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            w_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "z-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "z+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.1, t_end=0.1),
        ),
        assert_result=_assert_immunotherapy_3d,
    ),
    RuntimePresetCase(
        name="immunotherapy_2d_periodic_xy",
        make_config=lambda tmp_path: make_immunotherapy_config(
            tmp_path,
            gdim=2,
            u_initial_condition=scalar_expr("gaussian_noise", mean=0.3, std=0.05),
            v_initial_condition=scalar_expr("gaussian_noise", mean=0.9, std=0.05),
            w_initial_condition=scalar_expr("gaussian_noise", mean=0.01, std=0.002),
            u_boundary_conditions={},
            v_boundary_conditions={},
            w_boundary_conditions={},
            u_periodic_axes=(0, 1),
            v_periodic_axes=(0, 1),
            w_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.1, t_end=0.1),
        ),
        assert_result=_assert_immunotherapy_periodic,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="immunotherapy_2d_mixed_bc",
        make_config=lambda tmp_path: make_immunotherapy_config(
            tmp_path,
            gdim=2,
            u_initial_condition=scalar_expr("gaussian_noise", mean=0.3, std=0.05),
            v_initial_condition=scalar_expr("gaussian_noise", mean=0.9, std=0.05),
            w_initial_condition=scalar_expr("gaussian_noise", mean=0.01, std=0.002),
            u_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            v_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            w_boundary_conditions={
                "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            },
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.1, t_end=0.1),
        ),
        assert_result=_assert_immunotherapy_mixed_bc,
    ),
    RuntimePresetCase(
        name="van_der_pol_2d_periodic_xy",
        make_config=lambda tmp_path: make_van_der_pol_config(
            tmp_path,
            gdim=2,
            u_initial_condition=scalar_expr(
                "sine_waves",
                background=0.0,
                modes=[
                    {"amplitude": 0.03, "cycles": [1, 2], "phase": 0.0},
                    {"amplitude": -0.02, "cycles": [2, 1], "phase": 1.0},
                    {"amplitude": 0.01, "cycles": [3, 2], "phase": 0.5},
                ],
            ),
            v_initial_condition=constant(0.0),
            u_boundary_conditions={},
            v_boundary_conditions={},
            u_periodic_axes=(0, 1),
            v_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_van_der_pol_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="van_der_pol_3d_periodic_xyz",
        make_config=lambda tmp_path: make_van_der_pol_config(
            tmp_path,
            gdim=3,
            u_initial_condition=scalar_expr(
                "sine_waves",
                background=0.0,
                modes=[
                    {"amplitude": 0.03, "cycles": [1, 2, 1], "phase": 0.0},
                    {"amplitude": -0.02, "cycles": [2, 1, 2], "phase": 1.0},
                    {"amplitude": 0.01, "cycles": [3, 2, 1], "phase": 0.5},
                ],
            ),
            v_initial_condition=constant(0.0),
            u_boundary_conditions={},
            v_boundary_conditions={},
            u_periodic_axes=(0, 1, 2),
            v_periodic_axes=(0, 1, 2),
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_van_der_pol_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="van_der_pol_2d_field_specific_scalar_boundaries",
        make_config=lambda tmp_path: make_van_der_pol_config(
            tmp_path,
            gdim=2,
            u_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=0.0,
                std=0.05,
            ),
            v_initial_condition=constant(0.0),
            u_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            },
            v_boundary_conditions={
                "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            },
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_van_der_pol_mixed_bc,
    ),
    RuntimePresetCase(
        name="superlattice_2d_periodic_xy",
        make_config=lambda tmp_path: make_superlattice_config(
            tmp_path,
            gdim=2,
            u_1_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=3.0,
                std=0.1,
            ),
            v_1_initial_condition=constant(3.0),
            u_2_initial_condition=constant(3.0),
            v_2_initial_condition=constant(10.0),
            u_1_boundary_conditions={},
            v_1_boundary_conditions={},
            u_2_boundary_conditions={},
            v_2_boundary_conditions={},
            u_1_periodic_axes=(0, 1),
            v_1_periodic_axes=(0, 1),
            u_2_periodic_axes=(0, 1),
            v_2_periodic_axes=(0, 1),
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_superlattice_2d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="superlattice_3d_periodic_xyz",
        make_config=lambda tmp_path: make_superlattice_config(
            tmp_path,
            gdim=3,
            u_1_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=3.0,
                std=0.1,
            ),
            v_1_initial_condition=constant(3.0),
            u_2_initial_condition=constant(3.0),
            v_2_initial_condition=constant(10.0),
            u_1_boundary_conditions={},
            v_1_boundary_conditions={},
            u_2_boundary_conditions={},
            v_2_boundary_conditions={},
            u_1_periodic_axes=(0, 1, 2),
            v_1_periodic_axes=(0, 1, 2),
            u_2_periodic_axes=(0, 1, 2),
            v_2_periodic_axes=(0, 1, 2),
            mesh_resolution=(5, 5, 5),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.01, t_end=0.01),
        ),
        assert_result=_assert_superlattice_3d,
        skip_reason=skip_without_mpc,
    ),
    RuntimePresetCase(
        name="superlattice_2d_mixed_scalar_boundaries",
        make_config=lambda tmp_path: make_superlattice_config(
            tmp_path,
            gdim=2,
            u_1_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=3.0,
                std=0.05,
            ),
            v_1_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=3.0,
                std=0.05,
            ),
            u_2_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=3.0,
                std=0.05,
            ),
            v_2_initial_condition=scalar_expr(
                "gaussian_noise",
                mean=10.0,
                std=0.05,
            ),
            u_1_boundary_conditions=_mixed_scalar_boundary_conditions(),
            v_1_boundary_conditions={
                "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(
                    type="robin",
                    value=constant(0.0),
                    alpha=1.0,
                ),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(3.0)),
            },
            u_2_boundary_conditions={
                "x-": BoundaryConditionConfig(type="dirichlet", value=constant(3.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(
                    type="robin",
                    value=constant(0.05),
                    alpha=0.75,
                ),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(3.0)),
            },
            v_2_boundary_conditions={
                "x-": BoundaryConditionConfig(type="dirichlet", value=constant(10.0)),
                "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                "y-": BoundaryConditionConfig(
                    type="robin",
                    value=constant(0.0),
                    alpha=0.5,
                ),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(10.0)),
            },
            mesh_resolution=(8, 8),
            output_resolution=(4, 4),
            time=TimeConfig(dt=0.005, t_end=0.005),
        ),
        assert_result=_assert_superlattice_mixed_bc,
    ),
    RuntimePresetCase(
        name="maxwell_mixed_boundaries",
        make_config=lambda tmp_path: make_maxwell_config(
            tmp_path,
            boundary_conditions=_mixed_maxwell_boundary_conditions(),
            source=_maxwell_source(),
        ),
        assert_result=_assert_maxwell_case,
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
        name="plate_3d_domain_rejected",
        make_config=lambda tmp_path: make_plate_config(
            tmp_path,
            gdim=3,
            boundary_conditions=_simply_supported_plate_boundary_conditions(gdim=3),
            deflection_initial_condition=constant(0.0),
            velocity_initial_condition=constant(0.0),
            load_source=scalar_expr("none"),
            mesh_resolution=(4, 4, 4),
            output_resolution=(3, 3, 3),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        expected_error=ValueError,
        expected_error_match="only supports 2D domains",
    ),
    RuntimePresetCase(
        name="plate_unsupported_operator_rejected",
        make_config=lambda tmp_path: make_plate_config(
            tmp_path,
            gdim=2,
            boundary_conditions={
                **_simply_supported_plate_boundary_conditions(gdim=2),
                "x-": BoundaryConditionConfig(type="clamped"),
            },
            deflection_initial_condition=constant(0.0),
            velocity_initial_condition=constant(0.0),
            load_source=scalar_expr("none"),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        expected_error=ValueError,
        expected_error_match="unsupported operator 'clamped'",
    ),
    RuntimePresetCase(
        name="plate_missing_side_rejected",
        make_config=lambda tmp_path: make_plate_config(
            tmp_path,
            gdim=2,
            boundary_conditions={
                "x-": BoundaryConditionConfig(type="simply_supported"),
                "x+": BoundaryConditionConfig(type="simply_supported"),
                "y-": BoundaryConditionConfig(type="simply_supported"),
            },
            deflection_initial_condition=constant(0.0),
            velocity_initial_condition=constant(0.0),
            load_source=scalar_expr("none"),
            time=TimeConfig(dt=0.05, t_end=0.05),
        ),
        expected_error=ValueError,
        expected_error_match="must configure exactly the domain sides",
    ),
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
    RuntimePresetCase(
        name="shallow_water_nonperiodic_height_rejected",
        make_config=lambda tmp_path: make_shallow_water_config(
            tmp_path,
            parameters={
                "gravity": 1.0,
                "mean_depth": 1.0,
                "drag": 0.01,
                "viscosity": 0.002,
                "coriolis": 0.0,
            },
            bathymetry=constant(0.0),
            initial_height=constant(0.0),
            initial_velocity=vector_zero(),
            height_boundary_conditions={
                "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "x+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
                "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            },
            velocity_boundary_conditions={},
            velocity_periodic_axes=(0, 1),
            time=TimeConfig(dt=0.02, t_end=0.02),
        ),
        expected_error=ValueError,
        expected_error_match="unsupported operator 'dirichlet'",
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
