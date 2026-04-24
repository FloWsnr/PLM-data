"""Sampling and validation metadata for the rectangle domain."""

from plm_data.domains.base import (
    CoordinateSample,
    CoordinateRegionSampler,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_rectangle_params


def _uniform(context, name: str, minimum: float, maximum: float) -> float:
    value = float(context.child(name).rng.uniform(minimum, maximum))
    context.values[name] = value
    return value


def _box_sampler(
    region: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> CoordinateRegionSampler:
    def _sample(context, domain) -> CoordinateSample:
        length, height = domain.params["size"]
        return CoordinateSample(
            point=[
                _uniform(
                    context,
                    f"domain.region.{region}.x",
                    x_range[0] * length,
                    x_range[1] * length,
                ),
                _uniform(
                    context,
                    f"domain.region.{region}.y",
                    y_range[0] * height,
                    y_range[1] * height,
                ),
            ],
            scale=min(float(length), float(height)),
        )

    return _sample


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="rectangle",
        dimension=2,
        description="Axis-aligned 2D rectangle with four named side boundaries.",
        parameters={
            "size": DomainParameterSpec(
                name="size",
                kind="float_vector",
                length=2,
                hard_min=0.0,
                sampling_min=0.5,
                sampling_max=4.0,
                description="Domain side lengths [Lx, Ly].",
            ),
            "mesh_resolution": DomainParameterSpec(
                name="mesh_resolution",
                kind="int_vector",
                length=2,
                hard_min=1,
                sampling_min=32,
                sampling_max=256,
                description="Number of cells along each axis.",
            ),
        },
        boundary_names=("x-", "x+", "y-", "y+"),
        boundary_roles={
            "all": ("x-", "x+", "y-", "y+"),
            "x_min": ("x-",),
            "x_max": ("x+",),
            "y_min": ("y-",),
            "y_max": ("y+",),
            "x_pair": ("x-", "x+"),
            "y_pair": ("y-", "y+"),
            "walls": ("x-", "x+", "y-", "y+"),
        },
        periodic_pairs=(("x-", "x+"), ("y-", "y+")),
        supported_boundary_scenarios=(
            "scalar_all_neumann",
            "scalar_full_periodic",
            "plate_simply_supported",
            "schrodinger_zero_dirichlet",
            "burgers_zero_dirichlet",
            "elasticity_clamped_y",
            "darcy_pressure_drive",
            "fluid_velocity_no_slip",
            "shallow_water_full_periodic",
            "thermal_convection_rayleigh_benard",
            "maxwell_absorbing",
        ),
        supported_initial_condition_scenarios=(
            "advection_gaussian_bump",
            "wave_pulse",
            "plate_center_pulse",
            "schrodinger_wave_packet",
            "burgers_velocity_bump",
            "heat_gaussian_bump",
            "bistable_step_front",
            "brusselator_perturbed_state",
            "fitzhugh_nagumo_pulse",
            "gray_scott_spot",
            "gierer_meinhardt_perturbed",
            "schnakenberg_perturbed_state",
            "cyclic_competition_mixed",
            "immunotherapy_patch",
            "van_der_pol_modes",
            "lorenz_noisy_state",
            "keller_segel_cluster",
            "klausmeier_patch",
            "superlattice_perturbed_state",
            "cahn_hilliard_noise",
            "cgl_noisy_wave",
            "kuramoto_sivashinsky_modes",
            "swift_hohenberg_noise",
            "zakharov_kuznetsov_pulse",
            "fisher_step_front",
            "elasticity_transverse_mode",
            "darcy_tracer_blob",
            "navier_stokes_velocity_bump",
            "shallow_water_gaussian_height",
            "thermal_convection_layer",
            "maxwell_pulse_source",
        ),
        coordinate_regions=(
            "interior",
            "center",
            "left_half",
            "right_half",
            "lower_half",
            "upper_half",
        ),
        validate_params=validate_rectangle_params,
        coordinate_region_samplers={
            "interior": _box_sampler("interior", (0.25, 0.75), (0.25, 0.75)),
            "center": _box_sampler("center", (0.4, 0.6), (0.4, 0.6)),
            "left_half": _box_sampler("left_half", (0.2, 0.45), (0.25, 0.75)),
            "right_half": _box_sampler("right_half", (0.55, 0.8), (0.25, 0.75)),
            "lower_half": _box_sampler("lower_half", (0.25, 0.75), (0.2, 0.45)),
            "upper_half": _box_sampler("upper_half", (0.25, 0.75), (0.55, 0.8)),
        },
    )
)
