"""Tests for random runtime-config sampling."""

from dataclasses import is_dataclass, fields
from typing import Any

import pytest

from plm_data.pdes import get_pde, list_pdes
from plm_data.boundary_conditions.scenarios import (
    compatible_boundary_scenarios,
    list_boundary_scenarios,
)
from plm_data.initial_conditions.scenarios import (
    compatible_initial_condition_scenarios,
    list_initial_condition_scenarios,
)
from plm_data.sampling import (
    DEFAULT_RANDOM_OUTPUT_FORMATS,
    MIGRATED_RANDOM_PDES,
    list_random_pde_cases,
    sample_random_runtime_config,
    validate_runtime_config,
)


def _contains_sampler(value: Any) -> bool:
    if isinstance(value, dict):
        if "sample" in value:
            return True
        return any(_contains_sampler(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_sampler(item) for item in value)
    if is_dataclass(value):
        return any(
            _contains_sampler(getattr(value, field.name)) for field in fields(value)
        )
    return False


def test_pdes_registry_exposes_migrated_random_slice():
    pdes = list_pdes()

    assert tuple(pdes) == MIGRATED_RANDOM_PDES
    assert get_pde("wave").spec.name == "wave"
    assert get_pde("plate").spec.name == "plate"
    assert get_pde("schrodinger").spec.name == "schrodinger"
    assert get_pde("burgers").spec.name == "burgers"
    assert get_pde("bistable_travelling_waves").spec.name == "bistable_travelling_waves"
    assert get_pde("brusselator").spec.name == "brusselator"
    assert get_pde("fitzhugh_nagumo").spec.name == "fitzhugh_nagumo"
    assert get_pde("gray_scott").spec.name == "gray_scott"
    assert get_pde("gierer_meinhardt").spec.name == "gierer_meinhardt"
    assert get_pde("schnakenberg").spec.name == "schnakenberg"
    assert get_pde("cyclic_competition").spec.name == "cyclic_competition"
    assert get_pde("immunotherapy").spec.name == "immunotherapy"
    assert get_pde("van_der_pol").spec.name == "van_der_pol"
    assert get_pde("lorenz").spec.name == "lorenz"
    assert get_pde("keller_segel").spec.name == "keller_segel"
    assert get_pde("klausmeier_topography").spec.name == "klausmeier_topography"
    assert get_pde("superlattice").spec.name == "superlattice"
    assert get_pde("cahn_hilliard").spec.name == "cahn_hilliard"
    assert get_pde("cgl").spec.name == "cgl"
    assert get_pde("kuramoto_sivashinsky").spec.name == "kuramoto_sivashinsky"
    assert get_pde("swift_hohenberg").spec.name == "swift_hohenberg"
    assert get_pde("zakharov_kuznetsov").spec.name == "zakharov_kuznetsov"
    assert get_pde("heat").spec.name == "heat"
    assert get_pde("navier_stokes").spec.name == "navier_stokes"
    assert get_pde("shallow_water").spec.name == "shallow_water"
    assert get_pde("thermal_convection").spec.name == "thermal_convection"
    assert get_pde("maxwell_pulse").spec.name == "maxwell_pulse"
    with pytest.raises(ValueError, match="Unknown migrated PDE"):
        get_pde("not_migrated")


def test_domain_registry_module_exposes_migrated_layout():
    from plm_data.domains.registry import get_domain_spec, list_domain_specs

    specs = list_domain_specs()

    assert "rectangle" in specs
    assert get_domain_spec("rectangle").dimension == 2
    assert (
        "scalar_all_neumann"
        in get_domain_spec("rectangle").supported_boundary_scenarios
    )
    assert get_domain_spec("disk").supported_initial_condition_scenarios == (
        "heat_gaussian_bump",
    )


def test_domain_coordinate_region_samplers_are_executable():
    import math

    from plm_data.core.runtime_config import DomainConfig
    from plm_data.domains import sample_coordinate_region
    from plm_data.sampling import SamplingContext, attempt_rng

    context = SamplingContext(seed=17, attempt=0, rng=attempt_rng(17, 0, "test"))
    rectangle = DomainConfig(
        type="rectangle",
        params={"size": [2.0, 1.0], "mesh_resolution": [16, 8]},
    )
    rectangle_sample = sample_coordinate_region(rectangle, "left_half", context)
    assert 0.4 <= rectangle_sample.point[0] <= 0.9
    assert 0.25 <= rectangle_sample.point[1] <= 0.75
    assert rectangle_sample.scale == 1.0

    disk = DomainConfig(
        type="disk",
        params={"center": [1.0, -1.0], "radius": 2.0, "mesh_size": 0.1},
    )
    disk_sample = sample_coordinate_region(disk, "interior", context)
    disk_distance = math.hypot(disk_sample.point[0] - 1.0, disk_sample.point[1] + 1.0)
    assert disk_distance <= 1.1
    assert disk_sample.scale == 2.0

    annulus = DomainConfig(
        type="annulus",
        params={
            "center": [0.0, 0.0],
            "inner_radius": 0.5,
            "outer_radius": 1.5,
            "mesh_size": 0.1,
        },
    )
    annulus_sample = sample_coordinate_region(annulus, "interior", context)
    annulus_distance = math.hypot(annulus_sample.point[0], annulus_sample.point[1])
    assert 0.75 <= annulus_distance <= 1.25
    assert annulus_sample.scale == 1.0


def test_migrated_pde_packages_expose_specs_and_runtimes():
    from plm_data.pdes.advection import AdvectionPDE, PDE_SPEC as advection_spec
    from plm_data.pdes.bistable_travelling_waves import (
        BistableTravellingWavesPDE,
        PDE_SPEC as bistable_spec,
    )
    from plm_data.pdes.brusselator import BrusselatorPDE, PDE_SPEC as brusselator_spec
    from plm_data.pdes.burgers import BurgersPDE, PDE_SPEC as burgers_spec
    from plm_data.pdes.cahn_hilliard import (
        CahnHilliardPDE,
        PDE_SPEC as cahn_spec,
    )
    from plm_data.pdes.cgl import CGLPDE, PDE_SPEC as cgl_spec
    from plm_data.pdes.cyclic_competition import (
        CyclicCompetitionPDE,
        PDE_SPEC as cyclic_spec,
    )
    from plm_data.pdes.darcy import DarcyPDE, PDE_SPEC as darcy_spec
    from plm_data.pdes.elasticity import ElasticityPDE, PDE_SPEC as elasticity_spec
    from plm_data.pdes.fisher_kpp import FisherKPPPDE, PDE_SPEC as fisher_spec
    from plm_data.pdes.fitzhugh_nagumo import (
        FitzHughNagumoPDE,
        PDE_SPEC as fitzhugh_spec,
    )
    from plm_data.pdes.gierer_meinhardt import (
        GiererMeinhardtPDE,
        PDE_SPEC as gierer_spec,
    )
    from plm_data.pdes.gray_scott import GrayScottPDE, PDE_SPEC as gray_scott_spec
    from plm_data.pdes.heat import HeatPDE, PDE_SPEC as heat_spec
    from plm_data.pdes.immunotherapy import (
        ImmunotherapyPDE,
        PDE_SPEC as immunotherapy_spec,
    )
    from plm_data.pdes.keller_segel import KellerSegelPDE, PDE_SPEC as keller_spec
    from plm_data.pdes.klausmeier_topography import (
        KlausmeierTopographyPDE,
        PDE_SPEC as klausmeier_spec,
    )
    from plm_data.pdes.kuramoto_sivashinsky import (
        KuramotoSivashinskyPDE,
        PDE_SPEC as ks_spec,
    )
    from plm_data.pdes.lorenz import LorenzPDE, PDE_SPEC as lorenz_spec
    from plm_data.pdes.maxwell_pulse import (
        MaxwellPulsePDE,
        PDE_SPEC as maxwell_spec,
    )
    from plm_data.pdes.navier_stokes import (
        NavierStokesPDE,
        PDE_SPEC as navier_spec,
    )
    from plm_data.pdes.plate import PlatePDE, PDE_SPEC as plate_spec
    from plm_data.pdes.schrodinger import (
        PDE_SPEC as schrodinger_spec,
    )
    from plm_data.pdes.schrodinger import (
        SchrodingerPDE,
    )
    from plm_data.pdes.schnakenberg import (
        PDE_SPEC as schnakenberg_spec,
    )
    from plm_data.pdes.schnakenberg import (
        SchnakenbergPDE,
    )
    from plm_data.pdes.shallow_water import (
        PDE_SPEC as shallow_water_spec,
    )
    from plm_data.pdes.shallow_water import ShallowWaterPDE
    from plm_data.pdes.superlattice import (
        PDE_SPEC as superlattice_spec,
    )
    from plm_data.pdes.superlattice import SuperlatticePDE
    from plm_data.pdes.swift_hohenberg import (
        PDE_SPEC as swift_spec,
    )
    from plm_data.pdes.swift_hohenberg import SwiftHohenbergPDE
    from plm_data.pdes.thermal_convection import (
        PDE_SPEC as thermal_spec,
    )
    from plm_data.pdes.thermal_convection import ThermalConvectionPDE
    from plm_data.pdes.van_der_pol import PDE_SPEC as van_der_pol_spec
    from plm_data.pdes.van_der_pol import VanDerPolPDE
    from plm_data.pdes.wave import WavePDE, PDE_SPEC as wave_spec
    from plm_data.pdes.zakharov_kuznetsov import (
        PDE_SPEC as zk_spec,
    )
    from plm_data.pdes.zakharov_kuznetsov import ZakharovKuznetsovPDE

    assert AdvectionPDE().spec.name == advection_spec.name == "advection"
    assert WavePDE().spec.name == wave_spec.name == "wave"
    assert PlatePDE().spec.name == plate_spec.name == "plate"
    assert SchrodingerPDE().spec.name == schrodinger_spec.name == "schrodinger"
    assert BurgersPDE().spec.name == burgers_spec.name == "burgers"
    assert (
        BistableTravellingWavesPDE().spec.name
        == bistable_spec.name
        == "bistable_travelling_waves"
    )
    assert BrusselatorPDE().spec.name == brusselator_spec.name == "brusselator"
    assert FitzHughNagumoPDE().spec.name == fitzhugh_spec.name == "fitzhugh_nagumo"
    assert GrayScottPDE().spec.name == gray_scott_spec.name == "gray_scott"
    assert GiererMeinhardtPDE().spec.name == gierer_spec.name == "gierer_meinhardt"
    assert SchnakenbergPDE().spec.name == schnakenberg_spec.name == "schnakenberg"
    assert CyclicCompetitionPDE().spec.name == cyclic_spec.name == "cyclic_competition"
    assert ImmunotherapyPDE().spec.name == immunotherapy_spec.name == "immunotherapy"
    assert VanDerPolPDE().spec.name == van_der_pol_spec.name == "van_der_pol"
    assert LorenzPDE().spec.name == lorenz_spec.name == "lorenz"
    assert KellerSegelPDE().spec.name == keller_spec.name == "keller_segel"
    assert (
        KlausmeierTopographyPDE().spec.name
        == klausmeier_spec.name
        == "klausmeier_topography"
    )
    assert SuperlatticePDE().spec.name == superlattice_spec.name == "superlattice"
    assert CahnHilliardPDE().spec.name == cahn_spec.name == "cahn_hilliard"
    assert CGLPDE().spec.name == cgl_spec.name == "cgl"
    assert KuramotoSivashinskyPDE().spec.name == ks_spec.name == "kuramoto_sivashinsky"
    assert SwiftHohenbergPDE().spec.name == swift_spec.name == "swift_hohenberg"
    assert ZakharovKuznetsovPDE().spec.name == zk_spec.name == "zakharov_kuznetsov"
    assert HeatPDE().spec.name == heat_spec.name == "heat"
    assert ElasticityPDE().spec.name == elasticity_spec.name == "elasticity"
    assert FisherKPPPDE().spec.name == fisher_spec.name == "fisher_kpp"
    assert DarcyPDE().spec.name == darcy_spec.name == "darcy"
    assert NavierStokesPDE().spec.name == navier_spec.name == "navier_stokes"
    assert ShallowWaterPDE().spec.name == shallow_water_spec.name == "shallow_water"
    assert ThermalConvectionPDE().spec.name == thermal_spec.name == "thermal_convection"
    assert MaxwellPulsePDE().spec.name == maxwell_spec.name == "maxwell_pulse"


def test_random_profiles_cover_required_representative_slice():
    profiles = {profile.name: profile for profile in list_random_pde_cases()}
    expected_names = (
        "scalar_advection_rectangle",
        "scalar_wave_rectangle",
        "scalar_plate_rectangle",
        "split_schrodinger_rectangle",
        "vector_burgers_rectangle",
        "scalar_heat_2d",
        "nonlinear_bistable_travelling_waves_rectangle",
        "reaction_brusselator_rectangle",
        "reaction_fitzhugh_nagumo_rectangle",
        "reaction_gray_scott_rectangle",
        "reaction_gierer_meinhardt_rectangle",
        "reaction_schnakenberg_rectangle",
        "reaction_cyclic_competition_rectangle",
        "reaction_immunotherapy_rectangle",
        "oscillator_van_der_pol_rectangle",
        "chaotic_lorenz_rectangle",
        "chemotaxis_keller_segel_rectangle",
        "vegetation_klausmeier_topography_rectangle",
        "reaction_superlattice_rectangle",
        "phase_cahn_hilliard_rectangle",
        "oscillator_cgl_rectangle",
        "chaos_kuramoto_sivashinsky_rectangle",
        "pattern_swift_hohenberg_rectangle",
        "wave_zakharov_kuznetsov_rectangle",
        "vector_elasticity_rectangle",
        "nonlinear_fisher_kpp_rectangle",
        "fluid_darcy_rectangle",
        "fluid_navier_stokes_rectangle",
        "fluid_shallow_water_rectangle",
        "fluid_thermal_convection_rectangle",
        "electromagnetic_maxwell_pulse_rectangle",
    )

    assert tuple(profiles) == expected_names
    assert tuple(profile.pde_name for profile in list_random_pde_cases()) == (
        MIGRATED_RANDOM_PDES
    )
    assert set(profiles["scalar_heat_2d"].domain_samplers) == {
        "rectangle",
        "disk",
        "annulus",
    }

    for name, profile in profiles.items():
        if name != "scalar_heat_2d":
            assert set(profile.domain_samplers) == {"rectangle"}
        compatible_domains = [
            domain_name
            for domain_name in profile.domain_samplers
            if compatible_boundary_scenarios(
                pde_name=profile.pde_name,
                domain_name=domain_name,
            )
            and compatible_initial_condition_scenarios(
                pde_name=profile.pde_name,
                domain_name=domain_name,
            )
        ]
        assert compatible_domains


def test_boundary_scenarios_are_registered_and_compatible():
    scenarios = list_boundary_scenarios()

    assert "scalar_all_neumann" in scenarios
    assert "scalar_full_periodic" in scenarios
    assert "plate_simply_supported" in scenarios
    assert "schrodinger_zero_dirichlet" in scenarios
    assert "burgers_zero_dirichlet" in scenarios
    assert "elasticity_clamped_y" in scenarios
    assert "darcy_pressure_drive" in scenarios
    assert "fluid_velocity_no_slip" in scenarios
    assert "shallow_water_full_periodic" in scenarios
    assert "thermal_convection_rayleigh_benard" in scenarios
    assert "maxwell_absorbing" in scenarios
    assert scenarios["scalar_all_neumann"].spec.supported_dimensions == (2,)
    assert scenarios["scalar_all_neumann"].spec.required_boundary_roles == ("all",)
    assert scenarios["scalar_all_neumann"].spec.required_boundary_names == ()
    assert scenarios["scalar_full_periodic"].spec.operators == ("periodic",)
    assert scenarios["plate_simply_supported"].spec.operators == ("simply_supported",)
    assert scenarios["schrodinger_zero_dirichlet"].spec.configured_fields == ("u", "v")
    assert scenarios["burgers_zero_dirichlet"].spec.field_shapes == ("vector",)
    assert scenarios["elasticity_clamped_y"].spec.field_shapes == ("vector",)
    assert scenarios["maxwell_absorbing"].spec.operators == ("absorbing",)
    assert set(
        compatible_boundary_scenarios(pde_name="advection", domain_name="rectangle")
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(pde_name="wave", domain_name="rectangle")
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(pde_name="plate", domain_name="rectangle")
    ) == {"plate_simply_supported"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="schrodinger",
            domain_name="rectangle",
        )
    ) == {"schrodinger_zero_dirichlet"}
    assert set(
        compatible_boundary_scenarios(pde_name="burgers", domain_name="rectangle")
    ) == {"burgers_zero_dirichlet"}
    assert set(
        compatible_boundary_scenarios(pde_name="heat", domain_name="rectangle")
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="bistable_travelling_waves",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="brusselator",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="fitzhugh_nagumo",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="gray_scott",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="gierer_meinhardt",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="schnakenberg",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="cyclic_competition",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="immunotherapy",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="van_der_pol",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="lorenz",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="keller_segel",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="klausmeier_topography",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="superlattice",
            domain_name="rectangle",
        )
    ) == {"scalar_all_neumann"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="cahn_hilliard",
            domain_name="rectangle",
        )
    ) == {"scalar_full_periodic"}
    assert set(
        compatible_boundary_scenarios(pde_name="cgl", domain_name="rectangle")
    ) == {"scalar_full_periodic"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="kuramoto_sivashinsky",
            domain_name="rectangle",
        )
    ) == {"scalar_full_periodic"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="swift_hohenberg",
            domain_name="rectangle",
        )
    ) == {"scalar_full_periodic"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="zakharov_kuznetsov",
            domain_name="rectangle",
        )
    ) == {"scalar_full_periodic"}
    assert set(
        compatible_boundary_scenarios(pde_name="darcy", domain_name="rectangle")
    ) == {"darcy_pressure_drive"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="navier_stokes",
            domain_name="rectangle",
        )
    ) == {"fluid_velocity_no_slip"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="shallow_water",
            domain_name="rectangle",
        )
    ) == {"shallow_water_full_periodic"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="thermal_convection",
            domain_name="rectangle",
        )
    ) == {"thermal_convection_rayleigh_benard"}
    assert set(
        compatible_boundary_scenarios(
            pde_name="maxwell_pulse",
            domain_name="rectangle",
        )
    ) == {"maxwell_absorbing"}
    assert set(compatible_boundary_scenarios(pde_name="heat", domain_name="disk")) == {
        "scalar_all_neumann"
    }


def test_initial_condition_scenarios_are_registered_and_compatible():
    scenarios = list_initial_condition_scenarios()

    assert "advection_gaussian_bump" in scenarios
    assert "wave_pulse" in scenarios
    assert "plate_center_pulse" in scenarios
    assert "schrodinger_wave_packet" in scenarios
    assert "burgers_velocity_bump" in scenarios
    assert "heat_gaussian_bump" in scenarios
    assert "bistable_step_front" in scenarios
    assert "brusselator_perturbed_state" in scenarios
    assert "fitzhugh_nagumo_pulse" in scenarios
    assert "gray_scott_spot" in scenarios
    assert "gierer_meinhardt_perturbed" in scenarios
    assert "schnakenberg_perturbed_state" in scenarios
    assert "cyclic_competition_mixed" in scenarios
    assert "immunotherapy_patch" in scenarios
    assert "van_der_pol_modes" in scenarios
    assert "lorenz_noisy_state" in scenarios
    assert "keller_segel_cluster" in scenarios
    assert "klausmeier_patch" in scenarios
    assert "superlattice_perturbed_state" in scenarios
    assert "cahn_hilliard_noise" in scenarios
    assert "cgl_noisy_wave" in scenarios
    assert "kuramoto_sivashinsky_modes" in scenarios
    assert "swift_hohenberg_noise" in scenarios
    assert "zakharov_kuznetsov_pulse" in scenarios
    assert "fisher_step_front" in scenarios
    assert "elasticity_transverse_mode" in scenarios
    assert "darcy_tracer_blob" in scenarios
    assert "navier_stokes_velocity_bump" in scenarios
    assert "shallow_water_gaussian_height" in scenarios
    assert "thermal_convection_layer" in scenarios
    assert "maxwell_pulse_source" in scenarios
    assert scenarios["heat_gaussian_bump"].spec.supported_dimensions == (2,)
    assert scenarios["heat_gaussian_bump"].spec.field_shapes == ("scalar",)
    assert scenarios["darcy_tracer_blob"].spec.coordinate_regions == (
        "interior",
        "left_half",
    )
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="advection",
            domain_name="rectangle",
        )
    ) == {"advection_gaussian_bump"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="wave",
            domain_name="rectangle",
        )
    ) == {"wave_pulse"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="plate",
            domain_name="rectangle",
        )
    ) == {"plate_center_pulse"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="schrodinger",
            domain_name="rectangle",
        )
    ) == {"schrodinger_wave_packet"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="burgers",
            domain_name="rectangle",
        )
    ) == {"burgers_velocity_bump"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="elasticity",
            domain_name="rectangle",
        )
    ) == {"elasticity_transverse_mode"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="fisher_kpp",
            domain_name="rectangle",
        )
    ) == {"fisher_step_front"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="navier_stokes",
            domain_name="rectangle",
        )
    ) == {"navier_stokes_velocity_bump"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="shallow_water",
            domain_name="rectangle",
        )
    ) == {"shallow_water_gaussian_height"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="thermal_convection",
            domain_name="rectangle",
        )
    ) == {"thermal_convection_layer"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="maxwell_pulse",
            domain_name="rectangle",
        )
    ) == {"maxwell_pulse_source"}
    assert set(
        compatible_initial_condition_scenarios(pde_name="heat", domain_name="annulus")
    ) == {"heat_gaussian_bump"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="bistable_travelling_waves",
            domain_name="rectangle",
        )
    ) == {"bistable_step_front"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="brusselator",
            domain_name="rectangle",
        )
    ) == {"brusselator_perturbed_state"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="fitzhugh_nagumo",
            domain_name="rectangle",
        )
    ) == {"fitzhugh_nagumo_pulse"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="gray_scott",
            domain_name="rectangle",
        )
    ) == {"gray_scott_spot"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="gierer_meinhardt",
            domain_name="rectangle",
        )
    ) == {"gierer_meinhardt_perturbed"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="schnakenberg",
            domain_name="rectangle",
        )
    ) == {"schnakenberg_perturbed_state"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="cyclic_competition",
            domain_name="rectangle",
        )
    ) == {"cyclic_competition_mixed"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="immunotherapy",
            domain_name="rectangle",
        )
    ) == {"immunotherapy_patch"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="van_der_pol",
            domain_name="rectangle",
        )
    ) == {"van_der_pol_modes"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="lorenz",
            domain_name="rectangle",
        )
    ) == {"lorenz_noisy_state"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="keller_segel",
            domain_name="rectangle",
        )
    ) == {"keller_segel_cluster"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="klausmeier_topography",
            domain_name="rectangle",
        )
    ) == {"klausmeier_patch"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="superlattice",
            domain_name="rectangle",
        )
    ) == {"superlattice_perturbed_state"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="cahn_hilliard",
            domain_name="rectangle",
        )
    ) == {"cahn_hilliard_noise"}
    assert set(
        compatible_initial_condition_scenarios(pde_name="cgl", domain_name="rectangle")
    ) == {"cgl_noisy_wave"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="kuramoto_sivashinsky",
            domain_name="rectangle",
        )
    ) == {"kuramoto_sivashinsky_modes"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="swift_hohenberg",
            domain_name="rectangle",
        )
    ) == {"swift_hohenberg_noise"}
    assert set(
        compatible_initial_condition_scenarios(
            pde_name="zakharov_kuznetsov",
            domain_name="rectangle",
        )
    ) == {"zakharov_kuznetsov_pulse"}


def test_sample_random_runtime_config_is_concrete_and_2d():
    sampled = sample_random_runtime_config(seed=1234, attempt=0)

    assert sampled.pde_name in MIGRATED_RANDOM_PDES
    assert sampled.domain_name in {"rectangle", "disk", "annulus"}
    assert sampled.config.seed == 1234
    assert sampled.config.domain.dimension == 2
    assert sampled.config.time is not None
    assert sampled.config.output.formats == list(DEFAULT_RANDOM_OUTPUT_FORMATS)
    assert sampled.output_dir("/tmp/out").parts[-3:-1] == (
        sampled.pde_name,
        sampled.domain_name,
    )
    assert sampled.metadata()["boundary_scenario"]
    assert sampled.metadata()["initial_condition_scenario"]
    assert sampled.boundary_scenario_name in compatible_boundary_scenarios(
        pde_name=sampled.pde_name,
        domain_name=sampled.domain_name,
    )
    assert (
        sampled.initial_condition_scenario_name
        in compatible_initial_condition_scenarios(
            pde_name=sampled.pde_name,
            domain_name=sampled.domain_name,
        )
    )
    assert not _contains_sampler(sampled.config)
    validate_runtime_config(sampled.config)


def test_random_sampling_is_attempt_deterministic():
    sampled_a = sample_random_runtime_config(seed=42, attempt=1)
    sampled_b = sample_random_runtime_config(seed=42, attempt=1)
    sampled_c = sample_random_runtime_config(seed=42, attempt=2)

    assert sampled_a.metadata() == sampled_b.metadata()
    assert sampled_a.config == sampled_b.config
    assert (
        sampled_a.metadata() != sampled_c.metadata()
        or sampled_a.config != sampled_c.config
    )
