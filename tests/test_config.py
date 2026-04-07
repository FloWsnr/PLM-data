"""Tests for plm_data.core.config."""

from pathlib import Path

import pytest
import yaml

from plm_data.core.config import load_config
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_BLOCK_DIRECT,
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    NONLINEAR_MIXED_DIRECT,
    STATIONARY_INDEFINITE_DIRECT,
    STATIONARY_SCALAR_SPD,
    STEADY_SADDLE_POINT,
    TRANSIENT_MIXED_DIRECT,
    TRANSIENT_SADDLE_POINT,
)


def _solver_block(
    strategy: str,
    *,
    serial: dict[str, str] | None = None,
    mpi: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "serial": {"ksp_type": "preonly", "pc_type": "lu"}
        if serial is None
        else serial,
        "mpi": {"ksp_type": "preonly", "pc_type": "lu"} if mpi is None else mpi,
    }


def _write_yaml(tmp_path, name: str, data: dict[str, object]):
    path = tmp_path / name
    path.write_text(yaml.dump(data))
    return path


def test_load_config():
    cfg = load_config("configs/basic/heat/2d_localized_blob_diffusion.yaml")
    assert cfg.preset == "heat"
    assert cfg.parameters == {}
    assert cfg.coefficient("kappa").type == "constant"
    assert cfg.coefficient("kappa").params["value"] == 0.01
    assert cfg.domain.type == "rectangle"
    assert cfg.output.resolution == [64, 64]
    assert cfg.time.dt == 0.01
    assert cfg.time.t_end == 1.0
    assert cfg.has_periodic_boundary_conditions is False
    assert cfg.input("u").initial_condition.type == "gaussian_blobs"
    assert (
        cfg.input("u").initial_condition.params["generators"][0]["count"]["sample"]
        == "randint"
    )
    assert cfg.input("u").initial_condition.params["generators"][0]["count"]["min"] == 1
    assert cfg.input("u").initial_condition.params["generators"][0]["count"]["max"] == 2
    assert (
        cfg.input("u").initial_condition.params["generators"][0]["center"][0]["sample"]
        == "uniform"
    )
    assert cfg.input("u").source.type == "none"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "neumann"
    assert cfg.solver.strategy == CONSTANT_LHS_SCALAR_SPD
    assert cfg.solver.profile_name == "serial"
    assert cfg.solver.serial["pc_type"] == "lu"
    assert cfg.solver.mpi["pc_type"] == "hypre"


def test_load_config_periodic_field():
    cfg = load_config("configs/physics/cahn_hilliard/2d_spinodal_decomposition.yaml")
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.boundary_field("c").side_conditions("x-")[0].pair_with == "x+"
    assert cfg.boundary_field("c").side_conditions("y+")[0].type == "periodic"


def test_load_config_fisher_kpp_2d():
    cfg = load_config("configs/biology/fisher_kpp/2d_logistic_invasion_front.yaml")
    assert cfg.preset == "fisher_kpp"
    assert cfg.domain.dimension == 2
    assert cfg.parameters["D"] == 0.1
    assert cfg.parameters["r"] == 1.0
    assert cfg.parameters["K"] == 1.0
    assert cfg.coefficient("velocity").type == "zero"
    assert cfg.input("u").initial_condition.type == "step"
    assert cfg.input("u").initial_condition.params["x_split"]["sample"] == "uniform"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "neumann"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.solver.strategy == NONLINEAR_MIXED_DIRECT


def test_load_config_fisher_kpp_3d():
    cfg = load_config("configs/biology/fisher_kpp/3d_logistic_invasion_front.yaml")
    assert cfg.preset == "fisher_kpp"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is False
    assert cfg.coefficient("velocity").type == "zero"
    assert cfg.input("u").initial_condition.params["axis"] == 0
    assert cfg.input("u").initial_condition.params["x_split"]["sample"] == "uniform"
    assert cfg.boundary_field("u").side_conditions("z+")[0].type == "neumann"
    assert cfg.output.formats == ["numpy", "gif", "vtk"]
    assert cfg.solver.strategy == NONLINEAR_MIXED_DIRECT


def test_load_config_bistable_travelling_waves_2d():
    cfg = load_config(
        "configs/biology/bistable_travelling_waves/2d_planar_invasion_front.yaml"
    )
    assert cfg.preset == "bistable_travelling_waves"
    assert cfg.domain.dimension == 2
    assert cfg.parameters["D"] == 1.0
    assert cfg.parameters["a"] == 0.3
    assert cfg.coefficient("velocity").type == "zero"
    assert cfg.input("u").initial_condition.type == "step"
    assert cfg.input("u").initial_condition.params["value_left"] == 1.0
    assert cfg.boundary_field("u").side_conditions("y+")[0].type == "neumann"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.solver.strategy == NONLINEAR_MIXED_DIRECT


def test_load_config_bistable_travelling_waves_3d():
    cfg = load_config(
        "configs/biology/bistable_travelling_waves/3d_planar_invasion_front.yaml"
    )
    assert cfg.preset == "bistable_travelling_waves"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is False
    assert cfg.coefficient("velocity").type == "zero"
    assert cfg.input("u").initial_condition.params["x_split"]["sample"] == "uniform"
    assert cfg.boundary_field("u").side_conditions("z-")[0].type == "neumann"
    assert cfg.output.formats == ["numpy", "gif", "vtk"]
    assert cfg.solver.strategy == NONLINEAR_MIXED_DIRECT


def test_load_config_missing_field(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.dump({"parameters": {"k": 1.0}}))
    with pytest.raises(ValueError, match="preset"):
        load_config(bad_yaml)


def test_load_config_rejects_invalid_annulus_geometry(tmp_path):
    data = yaml.safe_load(
        Path(
            "configs/physics/gray_scott/2d_annular_spot_stripe_patterns.yaml"
        ).read_text()
    )
    data["domain"]["inner_radius"] = 1.25
    data["domain"]["outer_radius"] = 1.0

    path = _write_yaml(tmp_path, "bad_annulus.yaml", data)

    with pytest.raises(ValueError, match="inner_radius' < 'outer_radius"):
        load_config(path)


def test_load_config_boundary_field_sections():
    cfg = load_config("configs/basic/poisson/2d_sinusoidal_source_response.yaml")
    u_boundary = cfg.boundary_field("u")
    assert set(u_boundary.sides) == {"x-", "x+", "y-", "y+"}
    assert cfg.input("u").source.type == "sine_waves"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.solver.strategy == STATIONARY_SCALAR_SPD


def test_load_config_vector_input():
    cfg = load_config("configs/fluids/navier_stokes/2d_lid_driven_cavity_vortices.yaml")
    velocity = cfg.input("velocity")
    velocity_bcs = cfg.boundary_field("velocity")
    assert cfg.output_mode("velocity") == "components"
    assert cfg.output_mode("pressure") == "scalar"
    assert velocity.source.type == "none"
    assert velocity.initial_condition.components["x"].type == "gaussian_noise"
    assert velocity.initial_condition.components["x"].params["std"] == 0.03
    assert velocity.initial_condition.components["y"].params["std"] == 0.03
    assert (
        velocity_bcs.side_conditions("y+")[0].value.components["x"].params["value"]
        == 1.0
    )
    assert cfg.solver.profile_name == "serial"


def test_load_config_transient_saddle_point_uses_fieldsplit_mpi_profile():
    cfg = load_config("configs/fluids/navier_stokes/2d_lid_driven_cavity_vortices.yaml")

    assert cfg.solver.strategy == TRANSIENT_SADDLE_POINT
    assert cfg.solver.serial["pc_type"] == "lu"

    mpi_options = cfg.solver.options_for_size(4)
    assert mpi_options["pc_type"] == "fieldsplit"
    assert mpi_options["pc_fieldsplit_type"] == "schur"
    assert mpi_options["pc_fieldsplit_schur_fact_type"] == "upper"
    assert mpi_options["pc_fieldsplit_schur_precondition"] == "a11"


def test_load_config_navier_stokes_channel_obstacle_uses_transient_saddle_point():
    cfg = load_config(
        "configs/fluids/navier_stokes/2d_channel_obstacle_vortex_shedding.yaml"
    )

    assert cfg.solver.strategy == TRANSIENT_SADDLE_POINT
    inlet = cfg.boundary_field("velocity").side_conditions("inlet")[0]
    outlet = cfg.boundary_field("velocity").side_conditions("outlet")[0]
    assert inlet.type == "dirichlet"
    assert inlet.value.components["x"].type == "sine_waves"
    assert outlet.type == "neumann"


def test_load_config_thermal_convection_2d():
    cfg = load_config("configs/fluids/thermal_convection/2d_rayleigh_benard_rolls.yaml")
    assert cfg.preset == "thermal_convection"
    assert cfg.domain.dimension == 2
    assert cfg.output_mode("velocity") == "components"
    assert cfg.output_mode("pressure") == "scalar"
    assert cfg.output_mode("temperature") == "scalar"
    assert cfg.input("velocity").initial_condition.components["x"].type == "constant"
    assert (
        cfg.input("velocity").initial_condition.components["y"].params["value"] == 0.0
    )
    assert cfg.input("temperature").initial_condition.type == "gaussian_noise"
    assert cfg.input("temperature").initial_condition.params["mean"] == 0.5
    assert cfg.input("temperature").initial_condition.params["std"] == 0.15
    assert cfg.boundary_field("velocity").side_conditions("x-")[0].type == "periodic"
    assert (
        cfg.boundary_field("temperature").side_conditions("y-")[0].value.params["value"]
        == 1.0
    )
    assert cfg.solver.strategy == TRANSIENT_MIXED_DIRECT


def test_load_config_thermal_convection_3d():
    cfg = load_config(
        "configs/fluids/thermal_convection/3d_rayleigh_benard_plumes.yaml"
    )
    assert cfg.preset == "thermal_convection"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.input("temperature").initial_condition.type == "gaussian_noise"
    assert cfg.input("temperature").initial_condition.params["std"] == 0.18
    assert cfg.boundary_field("velocity").side_conditions("y+")[0].pair_with == "y-"
    assert (
        cfg.boundary_field("temperature").side_conditions("z+")[0].value.params["value"]
        == 0.0
    )


def test_load_config_gray_scott_2d():
    cfg = load_config(
        "configs/physics/gray_scott/2d_drifting_spot_stripe_patterns.yaml"
    )
    assert cfg.preset == "gray_scott"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["Du"] == 2.0e-5
    assert cfg.parameters["Dv"] == 1.0e-5
    assert cfg.parameters["F"] == 0.037
    assert cfg.parameters["k"] == 0.06
    assert cfg.coefficient("velocity").components["x"].params["value"] == 0.01
    assert cfg.coefficient("velocity").components["y"].params["value"] == 0.0
    assert cfg.input("u").initial_condition.type == "gaussian_blobs"
    assert cfg.input("v").initial_condition.type == "gaussian_blobs"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.output_mode("v") == "scalar"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "periodic"
    assert cfg.boundary_field("v").side_conditions("y+")[0].pair_with == "y-"


def test_load_config_gray_scott_3d():
    cfg = load_config("configs/physics/gray_scott/3d_spot_blob_patterns.yaml")
    assert cfg.preset == "gray_scott"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.coefficient("velocity").components["x"].params["value"] == 0.01
    assert cfg.coefficient("velocity").components["y"].params["value"] == 0.0
    assert cfg.coefficient("velocity").components["z"].params["value"] == 0.005
    assert len(cfg.input("u").initial_condition.params["generators"]) == 2
    assert (
        cfg.input("u").initial_condition.params["generators"][0]["center"][0]["sample"]
        == "uniform"
    )
    assert (
        cfg.input("v").initial_condition.params["generators"][1]["sigma"]["sample"]
        == "uniform"
    )
    assert cfg.boundary_field("u").side_conditions("z+")[0].pair_with == "z-"
    assert cfg.boundary_field("v").side_conditions("x-")[0].pair_with == "x+"


def test_load_config_fisher_kpp_disk_domain():
    cfg = load_config("configs/biology/fisher_kpp/2d_disk_radial_invasion.yaml")
    assert cfg.domain.type == "disk"
    assert cfg.domain.dimension == 2
    assert cfg.input("u").initial_condition.type == "gaussian_blobs"
    assert cfg.boundary_field("u").side_conditions("outer")[0].type == "neumann"


def test_load_config_keller_segel_parallelogram_domain():
    cfg = load_config(
        "configs/biology/keller_segel/2d_parallelogram_chemotactic_aggregation.yaml"
    )
    assert cfg.domain.type == "parallelogram"
    assert cfg.domain.dimension == 2
    assert cfg.boundary_field("rho").side_conditions("x-")[0].pair_with == "x+"
    assert cfg.boundary_field("c").side_conditions("y+")[0].pair_with == "y-"


def test_load_config_darcy_channel_obstacle_domain():
    cfg = load_config("configs/fluids/darcy/2d_channel_obstacle_tracer_transport.yaml")
    assert cfg.domain.type == "channel_obstacle"
    assert cfg.domain.dimension == 2
    assert set(cfg.boundary_field("pressure").sides) == {
        "inlet",
        "outlet",
        "walls",
        "obstacle",
    }
    assert (
        cfg.boundary_field("pressure").side_conditions("inlet")[0].type == "dirichlet"
    )
    assert (
        cfg.boundary_field("concentration").side_conditions("obstacle")[0].type
        == "neumann"
    )


def test_load_config_swift_hohenberg_2d_default():
    cfg = load_config("configs/physics/swift_hohenberg/2d_advected_roll_growth.yaml")
    assert cfg.preset == "swift_hohenberg"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["r"] == -0.28
    assert cfg.parameters["alpha"] == 1.6
    assert cfg.parameters["beta"] == -1.0
    assert cfg.parameters["gamma"] == -1.0
    assert cfg.coefficient("velocity").components["x"].params["value"] == 0.6
    assert cfg.coefficient("velocity").components["y"].params["value"] == -0.25
    assert cfg.input("u").initial_condition.type == "gaussian_blobs"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "periodic"


def test_load_config_swift_hohenberg_2d_rotational():
    cfg = load_config("configs/physics/swift_hohenberg/2d_rotating_roll_patterns.yaml")
    assert cfg.preset == "swift_hohenberg"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is False
    assert cfg.coefficient("velocity").components["x"].type == "affine"
    assert cfg.coefficient("velocity").components["y"].type == "affine"
    assert cfg.input("u").initial_condition.type == "gaussian_noise"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "simply_supported"


def test_load_config_swift_hohenberg_2d_directed():
    cfg = load_config("configs/physics/swift_hohenberg/2d_directed_roll_growth.yaml")
    assert cfg.preset == "swift_hohenberg"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.coefficient("velocity").components["x"].params["value"] == 0.9
    assert cfg.coefficient("velocity").components["y"].params["value"] == -0.4
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "periodic"


def test_load_config_swift_hohenberg_3d_default():
    cfg = load_config("configs/physics/swift_hohenberg/3d_advected_pattern_growth.yaml")
    assert cfg.preset == "swift_hohenberg"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["r"] == -0.28
    assert cfg.coefficient("velocity").components["z"].params["value"] == 0.15
    assert cfg.boundary_field("u").side_conditions("z+")[0].pair_with == "z-"


def test_load_config_van_der_pol_2d():
    cfg = load_config("configs/physics/van_der_pol/2d_oscillatory_wave_relaxation.yaml")
    assert cfg.preset == "van_der_pol"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["Du"] == 0.2
    assert cfg.parameters["Dv"] == 0.02
    assert cfg.parameters["mu"] == 4.0
    assert cfg.input("u").initial_condition.type == "sine_waves"
    assert cfg.input("u").initial_condition.params["background"] == 0.0
    assert len(cfg.input("u").initial_condition.params["modes"]) == 4
    assert cfg.input("v").initial_condition.type == "constant"
    assert cfg.input("v").initial_condition.params["value"] == 0.0
    assert cfg.output_mode("u") == "scalar"
    assert cfg.output_mode("v") == "scalar"
    assert cfg.boundary_field("u").side_conditions("x-")[0].pair_with == "x+"
    assert cfg.boundary_field("v").side_conditions("y+")[0].pair_with == "y-"


def test_load_config_van_der_pol_3d():
    cfg = load_config("configs/physics/van_der_pol/3d_oscillatory_wave_relaxation.yaml")
    assert cfg.preset == "van_der_pol"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.input("u").initial_condition.type == "sine_waves"
    assert len(cfg.input("u").initial_condition.params["modes"][0]["cycles"]) == 3
    assert cfg.input("u").initial_condition.params["modes"][0]["cycles"][0]["max"] == 4
    assert cfg.input("v").initial_condition.type == "constant"
    assert cfg.input("v").initial_condition.params["value"] == 0.0
    assert cfg.boundary_field("u").side_conditions("z+")[0].pair_with == "z-"
    assert cfg.boundary_field("v").side_conditions("x-")[0].pair_with == "x+"


def test_load_config_keller_segel_2d():
    cfg = load_config("configs/biology/keller_segel/2d_chemotactic_aggregation.yaml")
    assert cfg.preset == "keller_segel"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["chi0"] == 10.0
    assert cfg.input("rho").initial_condition.type == "gaussian_noise"
    assert cfg.input("c").initial_condition.params["mean"] == 1.0
    assert cfg.boundary_field("rho").side_conditions("x-")[0].type == "periodic"
    assert cfg.boundary_field("c").side_conditions("y+")[0].pair_with == "y-"
    assert cfg.output_mode("rho") == "scalar"
    assert cfg.output_mode("c") == "scalar"


def test_load_config_cyclic_competition_2d():
    cfg = load_config("configs/biology/cyclic_competition/2d_spatial_rps_domains.yaml")
    assert cfg.preset == "cyclic_competition"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["Du"] == 0.2
    assert cfg.parameters["a"] == 0.5
    assert cfg.parameters["b"] == 2.0
    assert cfg.input("u").initial_condition.type == "gaussian_noise"
    assert cfg.input("v").initial_condition.params["mean"] == 0.286
    assert cfg.input("w").initial_condition.params["std"] == 0.1
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "periodic"
    assert cfg.boundary_field("w").side_conditions("y+")[0].pair_with == "y-"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.output_mode("v") == "scalar"
    assert cfg.output_mode("w") == "scalar"


def test_load_config_shallow_water():
    cfg = load_config(
        "configs/fluids/shallow_water/2d_rotating_gravity_wave_pulse.yaml"
    )
    assert cfg.preset == "shallow_water"
    assert cfg.domain.dimension == 2
    assert cfg.parameters["gravity"] == 1.0
    assert cfg.parameters["mean_depth"] == 1.0
    assert cfg.coefficient("bathymetry").type == "constant"
    assert cfg.coefficient("bathymetry").params["value"] == 0.0
    assert cfg.input("height").initial_condition.type == "gaussian_blobs"
    assert (
        cfg.input("height").initial_condition.params["generators"][0]["count"]["sample"]
        == "randint"
    )
    assert (
        cfg.input("height").initial_condition.params["generators"][0]["count"]["min"]
        == 1
    )
    assert (
        cfg.input("height").initial_condition.params["generators"][0]["count"]["max"]
        == 2
    )
    assert cfg.input("velocity").initial_condition.components["x"].type == "constant"
    assert (
        cfg.input("velocity").initial_condition.components["y"].params["value"] == 0.0
    )
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.boundary_field("height").side_conditions("x-")[0].pair_with == "x+"
    assert cfg.boundary_field("velocity").side_conditions("y+")[0].pair_with == "y-"
    assert cfg.output_mode("height") == "scalar"
    assert cfg.output_mode("velocity") == "components"
    assert cfg.solver.strategy == NONLINEAR_MIXED_DIRECT


def test_load_config_advection_2d():
    cfg = load_config("configs/basic/advection/2d_cellular_blob_advection.yaml")
    assert cfg.preset == "advection"
    assert cfg.domain.dimension == 2
    assert cfg.output_mode("u") == "scalar"
    assert cfg.input("u").initial_condition.type == "gaussian_blobs"
    assert cfg.boundary_field("u").side_conditions("x-")[0].type == "periodic"
    assert cfg.coefficient("diffusivity").params["value"] == 0.0005
    velocity = cfg.coefficient("velocity")
    assert velocity.components["x"].type == "sine_waves"
    assert velocity.components["y"].params["modes"][0]["amplitude"] == -1.25
    assert cfg.solver.strategy == CONSTANT_LHS_SCALAR_NONSYMMETRIC
    assert cfg.solver.mpi["ksp_type"] == "gmres"


def test_load_config_advection_3d():
    cfg = load_config("configs/basic/advection/3d_swirling_blob_advection.yaml")
    assert cfg.preset == "advection"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.coefficient("velocity").components["z"].params["modes"][0]["cycles"] == [
        2.0,
        0.0,
        0.0,
    ]
    assert cfg.boundary_field("u").side_conditions("z+")[0].pair_with == "z-"


def test_load_config_burgers_2d():
    cfg = load_config("configs/fluids/burgers/2d_shear_mode_interaction.yaml")
    velocity = cfg.input("velocity")
    assert cfg.preset == "burgers"
    assert cfg.domain.dimension == 2
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["nu"] == 0.003
    assert cfg.output_mode("velocity") == "components"
    assert velocity.source.type == "none"
    assert velocity.initial_condition.components["x"].type == "sine_waves"
    assert velocity.initial_condition.components["x"].params["modes"][0]["cycles"] == [
        0,
        2,
    ]
    assert (
        velocity.initial_condition.components["y"].params["modes"][0]["amplitude"][
            "sample"
        ]
        == "uniform"
    )
    assert cfg.boundary_field("velocity").side_conditions("x-")[0].pair_with == "x+"
    assert cfg.solver.strategy == NONLINEAR_MIXED_DIRECT


def test_load_config_burgers_3d():
    cfg = load_config("configs/fluids/burgers/3d_shear_mode_interaction.yaml")
    velocity = cfg.input("velocity")
    assert cfg.preset == "burgers"
    assert cfg.domain.dimension == 3
    assert cfg.has_periodic_boundary_conditions is True
    assert cfg.parameters["nu"] == 0.004
    assert cfg.output_mode("velocity") == "components"
    assert velocity.initial_condition.components["y"].type == "sine_waves"
    assert velocity.initial_condition.components["y"].params["modes"][0]["cycles"] == [
        0,
        0,
        2,
    ]
    assert (
        velocity.initial_condition.components["z"].params["modes"][0]["amplitude"][
            "sample"
        ]
        == "uniform"
    )
    assert cfg.boundary_field("velocity").side_conditions("z+")[0].pair_with == "z-"
    assert cfg.solver.strategy == NONLINEAR_MIXED_DIRECT


def test_load_config_maxwell():
    cfg = load_config("configs/physics/maxwell/2d_localized_em_radiation.yaml")
    electric_field = cfg.input("electric_field")
    boundary_field = cfg.boundary_field("electric_field")
    assert cfg.output_mode("electric_field_real") == "components"
    assert cfg.output_mode("electric_field_imag") == "components"
    assert electric_field.source.components["x"].type == "gaussian_bump"
    assert boundary_field.side_conditions("x-")[0].type == "absorbing"
    assert cfg.solver.strategy == STATIONARY_INDEFINITE_DIRECT


def test_load_config_maxwell_pulse():
    cfg = load_config("configs/physics/maxwell_pulse/2d_guided_em_pulse.yaml")
    electric_field = cfg.input("electric_field")
    boundary_field = cfg.boundary_field("electric_field")
    assert cfg.output_mode("electric_field") == "components"
    assert electric_field.initial_condition.components["x"].type == "gaussian_noise"
    assert electric_field.initial_condition.components["y"].params["value"] == 0.0
    assert boundary_field.side_conditions("x-")[0].type == "absorbing"
    assert boundary_field.side_conditions("y-")[0].type == "dirichlet"
    assert cfg.solver.strategy == CONSTANT_LHS_CURL_DIRECT


def test_load_config_wave():
    cfg = load_config("configs/basic/wave/2d_localized_pulse_propagation.yaml")
    boundary_field = cfg.boundary_field("u")
    assert cfg.preset == "wave"
    assert cfg.parameters["damping"] == 0.01
    assert cfg.coefficient("c_sq").type == "radial_cosine"
    assert cfg.coefficient("c_sq").params["base"] == 1.8
    assert cfg.coefficient("c_sq").params["amplitude"] == 0.65
    assert cfg.input("u").initial_condition.type == "constant"
    assert cfg.input("u").initial_condition.params["value"] == 0.0
    assert cfg.input("v").initial_condition.type == "gaussian_blobs"
    assert (
        cfg.input("v").initial_condition.params["generators"][0]["count"]["sample"]
        == "randint"
    )
    assert cfg.input("v").initial_condition.params["generators"][0]["count"]["min"] == 1
    assert cfg.input("v").initial_condition.params["generators"][0]["count"]["max"] == 2
    assert cfg.input("forcing").source.type == "none"
    assert cfg.output_mode("u") == "scalar"
    assert cfg.output_mode("v") == "scalar"
    assert boundary_field.side_conditions("x-")[0].type == "neumann"
    assert boundary_field.side_conditions("y-")[0].type == "neumann"


def test_load_config_schrodinger():
    cfg = load_config("configs/basic/schrodinger/2d_wavepacket_barrier_scattering.yaml")
    boundary_field_u = cfg.boundary_field("u")
    boundary_field_v = cfg.boundary_field("v")
    assert cfg.preset == "schrodinger"
    assert cfg.parameters["D"] == 0.05
    assert cfg.parameters["theta"] == 0.5
    assert cfg.coefficient("potential").type == "gaussian_bump"
    assert cfg.coefficient("potential").params["amplitude"] == 8.0
    assert cfg.input("u").initial_condition.type == "gaussian_wave_packet"
    assert (
        cfg.input("u").initial_condition.params["wavevector"][0]["sample"] == "uniform"
    )
    assert cfg.input("v").initial_condition.params["phase"] == pytest.approx(
        -1.5707963267948966
    )
    assert cfg.output_mode("u") == "scalar"
    assert cfg.output_mode("v") == "scalar"
    assert cfg.output_mode("density") == "scalar"
    assert cfg.output_mode("potential") == "scalar"
    assert boundary_field_u.side_conditions("x-")[0].type == "dirichlet"
    assert boundary_field_v.side_conditions("y+")[0].type == "dirichlet"
    assert cfg.solver.strategy == CONSTANT_LHS_BLOCK_DIRECT


def test_load_config_schrodinger_3d():
    cfg = load_config("configs/basic/schrodinger/3d_wavepacket_barrier_scattering.yaml")
    assert cfg.domain.dimension == 3
    assert cfg.input("u").initial_condition.params["center"][0]["sample"] == "uniform"
    assert cfg.output.formats == ["numpy", "gif", "vtk"]
    assert cfg.boundary_field("u").side_conditions("z+")[0].type == "dirichlet"


def test_load_config_plate():
    cfg = load_config("configs/basic/plate/2d_simply_supported_mode_vibration.yaml")
    boundary_field = cfg.boundary_field("deflection")
    assert cfg.preset == "plate"
    assert cfg.parameters["theta"] == 0.5
    assert cfg.coefficient("rho_h").type == "constant"
    assert cfg.coefficient("damping").params["value"] == 0.0
    assert cfg.coefficient("rigidity").params["value"] == 0.2
    assert cfg.input("deflection").initial_condition.type == "sine_waves"
    assert (
        cfg.input("deflection").initial_condition.params["modes"][0]["amplitude"][
            "sample"
        ]
        == "uniform"
    )
    assert cfg.input("velocity").initial_condition.type == "constant"
    assert cfg.input("velocity").initial_condition.params["value"] == 0.0
    assert cfg.input("load").source.type == "none"
    assert cfg.output_mode("deflection") == "scalar"
    assert cfg.output_mode("velocity") == "scalar"
    assert boundary_field.side_conditions("x-")[0].type == "simply_supported"


def test_load_config_elasticity_2d():
    cfg = load_config("configs/basic/elasticity/2d_cantilever_impulse_ringdown.yaml")
    boundary_field = cfg.boundary_field("displacement")
    velocity = cfg.input("velocity")
    assert cfg.preset == "elasticity"
    assert cfg.domain.dimension == 2
    assert cfg.parameters["young_modulus"] == 6.0
    assert cfg.parameters["poisson_ratio"] == 0.3
    assert cfg.parameters["density"] == 1.0
    assert cfg.parameters["eta_mass"] == 0.02
    assert cfg.parameters["eta_stiffness"] == 0.002
    assert cfg.input("displacement").initial_condition.type == "zero"
    assert velocity.initial_condition.components["x"].params["value"] == 0.0
    assert velocity.initial_condition.components["y"].type == "gaussian_bump"
    assert (
        velocity.initial_condition.components["y"].params["center"][0]["sample"]
        == "uniform"
    )
    assert cfg.input("forcing").source.type == "zero"
    assert cfg.output_mode("displacement") == "components"
    assert cfg.output_mode("velocity") == "components"
    assert cfg.output_mode("von_mises") == "scalar"
    assert boundary_field.side_conditions("x-")[0].type == "dirichlet"
    assert boundary_field.side_conditions("y+")[0].type == "neumann"
    assert cfg.solver.strategy == CONSTANT_LHS_BLOCK_DIRECT


def test_load_config_elasticity_3d():
    cfg = load_config("configs/basic/elasticity/3d_cantilever_impulse_ringdown.yaml")
    boundary_field = cfg.boundary_field("displacement")
    velocity = cfg.input("velocity")
    assert cfg.preset == "elasticity"
    assert cfg.domain.dimension == 3
    assert cfg.output.resolution == [64, 16, 16]
    assert velocity.initial_condition.components["y"].type == "gaussian_bump"
    assert velocity.initial_condition.components["z"].params["value"] == 0.0
    assert boundary_field.side_conditions("x-")[0].type == "dirichlet"
    assert boundary_field.side_conditions("z+")[0].type == "neumann"
    assert cfg.solver.strategy == CONSTANT_LHS_BLOCK_DIRECT


def _base_yaml_dict():
    """Return a minimal valid config dict with all required top-level fields."""
    return {
        "preset": "poisson",
        "parameters": {"kappa": 1.0, "f_amplitude": 1.0},
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "inputs": {
            "u": {
                "source": {"type": "none", "params": {}},
            },
        },
        "boundary_conditions": {
            "u": {
                "x-": [{"operator": "dirichlet", "value": 0.0}],
                "x+": [{"operator": "dirichlet", "value": 0.0}],
                "y-": [{"operator": "dirichlet", "value": 0.0}],
                "y+": [{"operator": "dirichlet", "value": 0.0}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 1,
            "formats": ["numpy"],
            "fields": {"u": "scalar"},
        },
        "solver": {
            **_solver_block(STATIONARY_SCALAR_SPD),
        },
    }


def _base_heat_yaml_dict():
    return {
        "preset": "heat",
        "parameters": {},
        "coefficients": {
            "kappa": {"type": "constant", "params": {"value": 0.01}},
        },
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "time": {"dt": 0.01, "t_end": 0.01},
        "inputs": {
            "u": {
                "source": {"type": "none", "params": {}},
                "initial_condition": {
                    "type": "constant",
                    "params": {"value": 0.0},
                },
            }
        },
        "boundary_conditions": {
            "u": {
                "x-": [{"operator": "neumann", "value": 0.0}],
                "x+": [{"operator": "neumann", "value": 0.0}],
                "y-": [{"operator": "neumann", "value": 0.0}],
                "y+": [{"operator": "neumann", "value": 0.0}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 2,
            "formats": ["numpy"],
            "fields": {"u": "scalar"},
        },
        "solver": {
            **_solver_block(CONSTANT_LHS_SCALAR_SPD),
        },
        "seed": 42,
    }


def _base_stokes_yaml_dict():
    return {
        "preset": "stokes",
        "parameters": {"nu": 1.0},
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "inputs": {
            "velocity": {
                "source": {"type": "none", "params": {}},
            },
        },
        "boundary_conditions": {
            "velocity": {
                "x-": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "x+": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "y-": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "y+": [{"operator": "dirichlet", "value": [1.0, 0.0]}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 1,
            "formats": ["numpy"],
            "fields": {"velocity": "components", "pressure": "scalar"},
        },
        "solver": {
            **_solver_block(STEADY_SADDLE_POINT),
        },
    }


def _base_maxwell_pulse_yaml_dict():
    return {
        "preset": "maxwell_pulse",
        "parameters": {
            "epsilon_r": 1.0,
            "mu_r": 1.0,
            "sigma": 0.05,
            "pulse_amplitude": 1.0,
            "pulse_frequency": 4.0,
            "pulse_width": 0.1,
            "pulse_delay": 0.2,
        },
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "inputs": {
            "electric_field": {
                "source": {
                    "components": {
                        "x": {
                            "type": "gaussian_bump",
                            "params": {
                                "amplitude": 1.0,
                                "sigma": 0.1,
                                "center": [0.5, 0.5],
                            },
                        },
                        "y": {"type": "zero", "params": {}},
                    }
                },
                "initial_condition": {"type": "zero", "params": {}},
            }
        },
        "boundary_conditions": {
            "electric_field": {
                "x-": [
                    {"operator": "absorbing", "value": {"type": "zero", "params": {}}}
                ],
                "x+": [
                    {"operator": "absorbing", "value": {"type": "zero", "params": {}}}
                ],
                "y-": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
                "y+": [{"operator": "dirichlet", "value": [0.0, 0.0]}],
            }
        },
        "output": {
            "resolution": [8, 8],
            "num_frames": 2,
            "formats": ["numpy"],
            "fields": {"electric_field": "components"},
        },
        "solver": {
            **_solver_block(CONSTANT_LHS_CURL_DIRECT),
        },
        "time": {"dt": 0.05, "t_end": 0.1},
    }


def test_robin_bc_missing_alpha(tmp_path):
    data = _base_yaml_dict()
    data["boundary_conditions"]["u"]["x-"] = [{"operator": "robin", "value": 0.0}]
    p = _write_yaml(tmp_path, "robin_no_alpha.yaml", data)
    with pytest.raises(ValueError, match="alpha"):
        load_config(p)


def test_missing_inputs_section(tmp_path):
    data = _base_yaml_dict()
    del data["inputs"]
    p = _write_yaml(tmp_path, "no_inputs.yaml", data)
    with pytest.raises(ValueError, match="inputs"):
        load_config(p)


def test_missing_coefficients_section(tmp_path):
    data = _base_heat_yaml_dict()
    del data["coefficients"]
    p = _write_yaml(tmp_path, "no_coefficients.yaml", data)
    with pytest.raises(ValueError, match="coefficients"):
        load_config(p)


def test_missing_boundary_conditions_section(tmp_path):
    data = _base_yaml_dict()
    del data["boundary_conditions"]
    p = _write_yaml(tmp_path, "no_boundary_conditions.yaml", data)
    with pytest.raises(ValueError, match="boundary_conditions"):
        load_config(p)


def test_invalid_initial_condition_sampler_is_rejected(tmp_path):
    data = _base_heat_yaml_dict()
    data["inputs"]["u"]["initial_condition"] = {
        "type": "constant",
        "params": {
            "value": {
                "sample": "uniform",
                "min": 0.0,
                "upper": 1.0,
            }
        },
    }
    p = _write_yaml(tmp_path, "bad_ic_sampler.yaml", data)
    with pytest.raises(ValueError, match="uniform sampler"):
        load_config(p)


def test_invalid_quadrants_region_values_are_rejected(tmp_path):
    data = _base_heat_yaml_dict()
    data["inputs"]["u"]["initial_condition"] = {
        "type": "quadrants",
        "params": {
            "split": [0.5, 0.5],
            "region_values": {
                "00": 0.0,
                "01": 1.0,
                "10": 2.0,
            },
        },
    }
    p = _write_yaml(tmp_path, "bad_quadrants.yaml", data)
    with pytest.raises(ValueError, match="region_values must contain exactly"):
        load_config(p)


def test_missing_solver_strategy(tmp_path):
    data = _base_yaml_dict()
    del data["solver"]["strategy"]
    p = _write_yaml(tmp_path, "no_strategy.yaml", data)
    with pytest.raises(ValueError, match="Missing required field 'strategy' in solver"):
        load_config(p)


def test_missing_solver_profile(tmp_path):
    data = _base_yaml_dict()
    del data["solver"]["mpi"]
    p = _write_yaml(tmp_path, "no_solver_mpi.yaml", data)
    with pytest.raises(ValueError, match="Missing required field 'mpi' in solver"):
        load_config(p)


def test_missing_output_resolution(tmp_path):
    data = _base_yaml_dict()
    del data["output"]["resolution"]
    p = _write_yaml(tmp_path, "no_resolution.yaml", data)
    with pytest.raises(ValueError, match="resolution"):
        load_config(p)


def test_output_path_is_rejected(tmp_path):
    data = _base_yaml_dict()
    data["output"]["path"] = "./output"
    p = _write_yaml(tmp_path, "output_path.yaml", data)
    with pytest.raises(ValueError, match="unsupported keys"):
        load_config(p)


def test_periodic_pair_must_be_reciprocal(tmp_path):
    data = _base_yaml_dict()
    data["boundary_conditions"]["u"] = {
        "x-": [{"operator": "periodic", "pair_with": "x+"}],
        "x+": [{"operator": "periodic", "pair_with": "y+"}],
        "y-": [{"operator": "dirichlet", "value": 0.0}],
        "y+": [{"operator": "dirichlet", "value": 0.0}],
    }
    p = _write_yaml(tmp_path, "bad_periodic_pair.yaml", data)
    with pytest.raises(ValueError, match="reciprocal"):
        load_config(p)


def test_parse_domain_periodic_map(tmp_path):
    data = _base_yaml_dict()
    data["domain"]["periodic_maps"] = {
        "streamwise": {
            "slave": "x+",
            "master": "x-",
            "transform": {
                "type": "affine",
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
                "offset": [-1.0, 0.0],
            },
        }
    }
    p = _write_yaml(tmp_path, "periodic_map.yaml", data)

    cfg = load_config(p)
    assert "streamwise" in cfg.domain.periodic_maps
    assert cfg.domain.periodic_maps["streamwise"].slave == "x+"


def test_load_config_vector_neumann_bc(tmp_path):
    data = _base_stokes_yaml_dict()
    data["boundary_conditions"]["velocity"]["x-"] = [
        {
            "operator": "neumann",
            "value": [1.0, 0.0],
        }
    ]
    p = _write_yaml(tmp_path, "stokes_vector_neumann.yaml", data)

    cfg = load_config(p)
    bc = cfg.boundary_field("velocity").side_conditions("x-")[0]
    assert bc.type == "neumann"
    assert bc.value.components["x"].params["value"] == 1.0
    assert bc.value.components["y"].params["value"] == 0.0


def test_vector_robin_bc_rejected_at_parse_time(tmp_path):
    data = _base_stokes_yaml_dict()
    data["boundary_conditions"]["velocity"]["x-"] = [
        {
            "operator": "robin",
            "value": [1.0, 0.0],
            "operator_parameters": {"alpha": 1.0},
        }
    ]
    p = _write_yaml(tmp_path, "stokes_vector_robin.yaml", data)

    with pytest.raises(ValueError, match="unsupported operator"):
        load_config(p)


def test_load_config_vector_absorbing_bc(tmp_path):
    data = _base_maxwell_pulse_yaml_dict()
    p = _write_yaml(tmp_path, "maxwell_pulse_absorbing.yaml", data)

    cfg = load_config(p)
    bc = cfg.boundary_field("electric_field").side_conditions("x-")[0]
    assert bc.type == "absorbing"
    assert bc.value.type == "zero"


def test_load_config_resolves_mapping_fragment_ref(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {"$ref": "solver.profile.stationary_scalar_spd"}
    p = _write_yaml(tmp_path, "solver_ref.yaml", data)

    cfg = load_config(p)

    assert cfg.solver.strategy == STATIONARY_SCALAR_SPD
    assert cfg.solver.serial["pc_type"] == "lu"
    assert cfg.solver.mpi["pc_type"] == "hypre"


def test_load_config_resolves_list_and_scalar_fragment_refs(tmp_path):
    data = _base_yaml_dict()
    data["output"] = {
        "resolution": [8, 8],
        "num_frames": 1,
        "formats": {"$ref": "output.formats.standard"},
        "fields": {"u": {"$ref": "output.mode.scalar"}},
    }
    p = _write_yaml(tmp_path, "output_ref.yaml", data)

    cfg = load_config(p)

    assert cfg.output.formats == ["numpy", "gif", "vtk"]
    assert cfg.output_mode("u") == "scalar"


def test_load_config_fragment_override_deep_merges_mappings(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {
        "$ref": "solver.profile.stationary_scalar_spd",
        "mpi": {"ksp_rtol": 1.0e-8},
    }
    p = _write_yaml(tmp_path, "solver_override.yaml", data)

    cfg = load_config(p)

    assert cfg.solver.mpi["pc_type"] == "hypre"
    assert float(cfg.solver.mpi["ksp_rtol"]) == pytest.approx(1.0e-8)


def test_load_config_fragment_override_replaces_lists(tmp_path):
    data = _base_yaml_dict()
    data["output"] = {
        "$ref": "output.template.scalar_u",
        "resolution": [8, 8],
        "num_frames": 1,
        "formats": ["vtk"],
    }
    p = _write_yaml(tmp_path, "output_override.yaml", data)

    cfg = load_config(p)

    assert cfg.output.formats == ["vtk"]


def test_load_config_missing_fragment_ref(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {"$ref": "solver.profile.missing"}
    p = _write_yaml(tmp_path, "missing_ref.yaml", data)

    with pytest.raises(ValueError, match="unknown fragment 'solver.profile.missing'"):
        load_config(p)


def test_load_config_fragment_cycle_is_rejected(tmp_path):
    data = _base_yaml_dict()
    data["solver"] = {"$ref": "test.cycle.a"}
    p = _write_yaml(tmp_path, "cycle_ref.yaml", data)

    with pytest.raises(ValueError, match="Detected fragment reference cycle"):
        load_config(p)


def test_load_config_rejects_override_on_non_mapping_fragment(tmp_path):
    data = _base_yaml_dict()
    data["output"]["fields"]["u"] = {
        "$ref": "output.mode.scalar",
        "mode": "components",
    }
    p = _write_yaml(tmp_path, "bad_scalar_override.yaml", data)

    with pytest.raises(ValueError, match="cannot apply local overrides"):
        load_config(p)


def test_load_config_rejects_unknown_top_level_keys(tmp_path):
    data = _base_yaml_dict()
    data["unexpected"] = {"$ref": "output.mode.scalar"}
    p = _write_yaml(tmp_path, "bad_top_level.yaml", data)

    with pytest.raises(ValueError, match="unsupported top-level keys"):
        load_config(p)
