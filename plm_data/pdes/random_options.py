"""Random-run options attached to PDE specs."""

from math import tau

from plm_data.core.runtime_config import FieldExpressionConfig
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_BLOCK_DIRECT,
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    NONLINEAR_MIXED_DIRECT,
    TRANSIENT_MIXED_DIRECT,
)
from plm_data.sampling.specs import (
    RandomDomainConstraint,
    RandomOutputSpec,
    RandomPDEOptions,
    RandomTimeSpec,
)


def _uniform(context, name: str, minimum: float, maximum: float) -> float:
    from plm_data.sampling.samplers import uniform

    return uniform(context, name, minimum, maximum)


def _expr(expr_type: str, **params) -> FieldExpressionConfig:
    return FieldExpressionConfig(type=expr_type, params=params)


def _constant(value: float | str) -> FieldExpressionConfig:
    return _expr("constant", value=value)


def _vector_zero() -> FieldExpressionConfig:
    return FieldExpressionConfig(type="zero")


def _vector_expr(**components: FieldExpressionConfig) -> FieldExpressionConfig:
    return FieldExpressionConfig(components=components)


def _heat_coefficients(context, domain, parameters):
    return {"kappa": _constant(_uniform(context, "heat.kappa", 0.006, 0.014))}


def _advection_coefficients(context, domain, parameters):
    speed = _uniform(context, "advection.speed", 0.35, 0.75)
    angle = _uniform(context, "advection.angle", -0.25, 0.25)
    return {
        "velocity": _vector_expr(x=_constant(speed), y=_constant(speed * angle)),
        "diffusivity": _constant(
            _uniform(context, "advection.diffusivity", 0.0015, 0.004)
        ),
    }


def _wave_coefficients(context, domain, parameters):
    return {"c_sq": _constant(_uniform(context, "wave.c_sq", 0.45, 1.0))}


def _plate_coefficients(context, domain, parameters):
    return {
        "rho_h": _constant(_uniform(context, "plate.rho_h", 0.9, 1.2)),
        "damping": _constant(_uniform(context, "plate.damping", 0.04, 0.12)),
        "rigidity": _constant(_uniform(context, "plate.rigidity", 0.012, 0.035)),
    }


def _schrodinger_coefficients(context, domain, parameters):
    return {
        "potential": _constant(_uniform(context, "schrodinger.potential", -0.08, 0.08))
    }


def _zero_velocity(context, domain, parameters):
    return {"velocity": _vector_zero()}


def _klausmeier_coefficients(context, domain, parameters):
    return {
        "topography": _expr(
            "sine_waves",
            background=0.0,
            modes=[
                {
                    "amplitude": _uniform(context, "klausmeier.topo.amp0", 0.08, 0.18),
                    "cycles": [1.0, 0.4],
                    "phase": _uniform(context, "klausmeier.topo.phase0", 0.0, tau),
                    "angle": _uniform(context, "klausmeier.topo.angle0", -0.5, 0.5),
                },
                {
                    "amplitude": _uniform(context, "klausmeier.topo.amp1", -0.08, 0.08),
                    "cycles": [0.4, 0.9],
                    "phase": _uniform(context, "klausmeier.topo.phase1", 0.0, tau),
                    "angle": _uniform(context, "klausmeier.topo.angle1", -0.5, 0.5),
                },
            ],
        )
    }


def _darcy_coefficients(context, domain, parameters):
    height = domain.params["size"][1]
    return {
        "mobility": _expr(
            "step",
            value_left=_uniform(context, "darcy.mobility.left", 0.12, 0.2),
            value_right=_uniform(context, "darcy.mobility.right", 0.22, 0.34),
            x_split=_uniform(
                context,
                "darcy.mobility.split",
                0.35 * height,
                0.65 * height,
            ),
            axis=1,
        ),
        "dispersion": _constant(_uniform(context, "darcy.dispersion", 0.001, 0.002)),
    }


def _shallow_water_coefficients(context, domain, parameters):
    return {"bathymetry": _constant(0.0)}


def _rect(
    *,
    length: tuple[float, float],
    height: tuple[float, float],
    cells_x: tuple[int, int],
) -> tuple[RandomDomainConstraint, ...]:
    return (
        RandomDomainConstraint(
            domain="rectangle",
            params={"length": length, "height": height, "cells_x": cells_x},
        ),
    )


def _options(
    case_name: str,
    solver_strategy: str,
    *,
    dt: float,
    t_end: float,
    fields: dict[str, str],
    base_resolution: tuple[int, int],
    num_frames: int,
    resolution_jitter: tuple[int, int],
    domains: tuple[RandomDomainConstraint, ...],
    coefficient_sampler=None,
    allow_unconstrained_domains: bool = False,
) -> RandomPDEOptions:
    return RandomPDEOptions(
        case_name=case_name,
        solver_strategy=solver_strategy,
        time=RandomTimeSpec(dt=dt, t_end=t_end),
        output=RandomOutputSpec(
            fields=fields,
            base_resolution=base_resolution,
            num_frames=num_frames,
            resolution_jitter=resolution_jitter,
        ),
        domain_constraints=domains,
        coefficient_sampler=coefficient_sampler,
        allow_unconstrained_domains=allow_unconstrained_domains,
    )


_RANDOM_OPTIONS = {
    "advection": _options(
        "scalar_advection",
        CONSTANT_LHS_SCALAR_NONSYMMETRIC,
        dt=0.005,
        t_end=0.035,
        fields={"u": "scalar"},
        base_resolution=(64, 44),
        num_frames=8,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.2, 1.8), height=(0.8, 1.2), cells_x=(32, 44)),
        coefficient_sampler=_advection_coefficients,
    ),
    "wave": _options(
        "scalar_wave",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.004,
        t_end=0.032,
        fields={"u": "scalar", "v": "scalar"},
        base_resolution=(58, 50),
        num_frames=8,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.0, 1.4), height=(0.8, 1.2), cells_x=(30, 42)),
        coefficient_sampler=_wave_coefficients,
    ),
    "plate": _options(
        "scalar_plate",
        CONSTANT_LHS_BLOCK_DIRECT,
        dt=0.003,
        t_end=0.018,
        fields={"deflection": "scalar", "velocity": "scalar"},
        base_resolution=(58, 42),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.1, 1.6), height=(0.75, 1.05), cells_x=(24, 34)),
        coefficient_sampler=_plate_coefficients,
    ),
    "schrodinger": _options(
        "split_schrodinger",
        CONSTANT_LHS_BLOCK_DIRECT,
        dt=0.003,
        t_end=0.018,
        fields={
            "u": "scalar",
            "v": "scalar",
            "density": "scalar",
            "potential": "scalar",
        },
        base_resolution=(64, 44),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.3, 1.8), height=(0.8, 1.2), cells_x=(28, 40)),
        coefficient_sampler=_schrodinger_coefficients,
    ),
    "burgers": _options(
        "vector_burgers",
        NONLINEAR_MIXED_DIRECT,
        dt=0.004,
        t_end=0.024,
        fields={"velocity": "components"},
        base_resolution=(60, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.1, 1.6), height=(0.8, 1.2), cells_x=(28, 40)),
    ),
    "heat": _options(
        "scalar_heat_2d",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.01,
        t_end=0.08,
        fields={"u": "scalar"},
        base_resolution=(56, 56),
        num_frames=9,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(0.9, 1.3), height=(0.8, 1.2), cells_x=(28, 40)),
        coefficient_sampler=_heat_coefficients,
        allow_unconstrained_domains=True,
    ),
    "bistable_travelling_waves": _options(
        "nonlinear_bistable_travelling_waves",
        NONLINEAR_MIXED_DIRECT,
        dt=0.018,
        t_end=0.108,
        fields={"u": "scalar"},
        base_resolution=(64, 44),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(2.3, 3.2), height=(1.4, 2.0), cells_x=(28, 40)),
        coefficient_sampler=_zero_velocity,
    ),
    "brusselator": _options(
        "reaction_brusselator",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.004,
        t_end=0.024,
        fields={"u": "scalar", "v": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.8, 2.6), height=(1.2, 1.8), cells_x=(28, 40)),
    ),
    "fitzhugh_nagumo": _options(
        "reaction_fitzhugh_nagumo",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.004,
        t_end=0.024,
        fields={"u": "scalar", "v": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.6, 2.2), height=(1.0, 1.5), cells_x=(28, 40)),
    ),
    "gray_scott": _options(
        "reaction_gray_scott",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.004,
        t_end=0.024,
        fields={"u": "scalar", "v": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.8, 2.6), height=(1.2, 1.8), cells_x=(28, 40)),
        coefficient_sampler=_zero_velocity,
    ),
    "gierer_meinhardt": _options(
        "reaction_gierer_meinhardt",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.004,
        t_end=0.024,
        fields={"a": "scalar", "h": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.6, 2.4), height=(1.1, 1.7), cells_x=(28, 40)),
    ),
    "schnakenberg": _options(
        "reaction_schnakenberg",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.003,
        t_end=0.018,
        fields={"u": "scalar", "v": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.8, 2.6), height=(1.2, 1.8), cells_x=(28, 40)),
    ),
    "cyclic_competition": _options(
        "reaction_cyclic_competition",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.004,
        t_end=0.024,
        fields={"u": "scalar", "v": "scalar", "w": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.8, 2.6), height=(1.2, 1.8), cells_x=(28, 40)),
    ),
    "immunotherapy": _options(
        "reaction_immunotherapy",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.03,
        t_end=0.18,
        fields={"u": "scalar", "v": "scalar", "w": "scalar"},
        base_resolution=(64, 48),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(18.0, 24.0), height=(14.0, 20.0), cells_x=(28, 40)),
    ),
    "van_der_pol": _options(
        "oscillator_van_der_pol",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.006,
        t_end=0.036,
        fields={"u": "scalar", "v": "scalar"},
        base_resolution=(60, 60),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(3.0, 4.2), height=(3.0, 4.2), cells_x=(28, 40)),
    ),
    "lorenz": _options(
        "chaotic_lorenz",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.0015,
        t_end=0.009,
        fields={"x": "scalar", "y": "scalar", "z": "scalar"},
        base_resolution=(62, 62),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(7.2, 9.5), height=(7.0, 9.0), cells_x=(28, 40)),
    ),
    "keller_segel": _options(
        "chemotaxis_keller_segel",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.003,
        t_end=0.018,
        fields={"rho": "scalar", "c": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.6, 2.4), height=(1.1, 1.7), cells_x=(28, 40)),
    ),
    "klausmeier_topography": _options(
        "vegetation_klausmeier_topography",
        CONSTANT_LHS_SCALAR_NONSYMMETRIC,
        dt=0.003,
        t_end=0.018,
        fields={"w": "scalar", "n": "scalar", "topography": "scalar"},
        base_resolution=(64, 44),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(2.2, 3.0), height=(1.4, 2.0), cells_x=(28, 40)),
        coefficient_sampler=_klausmeier_coefficients,
    ),
    "superlattice": _options(
        "reaction_superlattice",
        CONSTANT_LHS_SCALAR_SPD,
        dt=0.003,
        t_end=0.018,
        fields={"u_1": "scalar", "v_1": "scalar", "u_2": "scalar", "v_2": "scalar"},
        base_resolution=(62, 46),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.8, 2.6), height=(1.2, 1.8), cells_x=(28, 40)),
    ),
    "cahn_hilliard": _options(
        "phase_cahn_hilliard",
        NONLINEAR_MIXED_DIRECT,
        dt=0.0015,
        t_end=0.006,
        fields={"c": "scalar"},
        base_resolution=(56, 44),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.2, 1.8), height=(1.0, 1.5), cells_x=(24, 34)),
    ),
    "cgl": _options(
        "oscillator_cgl",
        NONLINEAR_MIXED_DIRECT,
        dt=0.0015,
        t_end=0.006,
        fields={"u": "scalar", "v": "scalar", "amplitude": "scalar"},
        base_resolution=(56, 44),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.4, 2.0), height=(1.0, 1.5), cells_x=(24, 34)),
    ),
    "kuramoto_sivashinsky": _options(
        "chaos_kuramoto_sivashinsky",
        NONLINEAR_MIXED_DIRECT,
        dt=0.001,
        t_end=0.005,
        fields={"u": "scalar", "v": "scalar"},
        base_resolution=(58, 48),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(4.0, 5.5), height=(3.2, 4.8), cells_x=(24, 34)),
    ),
    "swift_hohenberg": _options(
        "pattern_swift_hohenberg",
        NONLINEAR_MIXED_DIRECT,
        dt=0.0015,
        t_end=0.006,
        fields={"u": "scalar"},
        base_resolution=(58, 48),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(4.0, 5.5), height=(3.2, 4.8), cells_x=(24, 34)),
        coefficient_sampler=_zero_velocity,
    ),
    "zakharov_kuznetsov": _options(
        "wave_zakharov_kuznetsov",
        NONLINEAR_MIXED_DIRECT,
        dt=0.0015,
        t_end=0.006,
        fields={"u": "scalar"},
        base_resolution=(58, 42),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(2.5, 3.5), height=(1.5, 2.2), cells_x=(24, 34)),
    ),
    "elasticity": _options(
        "vector_elasticity",
        CONSTANT_LHS_BLOCK_DIRECT,
        dt=0.004,
        t_end=0.024,
        fields={
            "displacement": "components",
            "velocity": "components",
            "von_mises": "scalar",
        },
        base_resolution=(64, 36),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.3, 1.8), height=(0.45, 0.7), cells_x=(28, 38)),
    ),
    "fisher_kpp": _options(
        "nonlinear_fisher_kpp",
        NONLINEAR_MIXED_DIRECT,
        dt=0.02,
        t_end=0.12,
        fields={"u": "scalar"},
        base_resolution=(64, 48),
        num_frames=8,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(2.6, 3.4), height=(1.6, 2.2), cells_x=(30, 42)),
        coefficient_sampler=_zero_velocity,
    ),
    "darcy": _options(
        "fluid_darcy",
        CONSTANT_LHS_SCALAR_NONSYMMETRIC,
        dt=0.005,
        t_end=0.03,
        fields={
            "pressure": "scalar",
            "concentration": "scalar",
            "velocity": "components",
            "speed": "scalar",
        },
        base_resolution=(64, 40),
        num_frames=7,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.8, 2.4), height=(0.9, 1.25), cells_x=(32, 44)),
        coefficient_sampler=_darcy_coefficients,
    ),
    "navier_stokes": _options(
        "fluid_navier_stokes",
        TRANSIENT_MIXED_DIRECT,
        dt=0.004,
        t_end=0.016,
        fields={"velocity": "components", "pressure": "scalar"},
        base_resolution=(56, 36),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.4, 2.0), height=(0.8, 1.2), cells_x=(18, 28)),
    ),
    "shallow_water": _options(
        "fluid_shallow_water",
        NONLINEAR_MIXED_DIRECT,
        dt=0.0015,
        t_end=0.006,
        fields={"height": "scalar", "velocity": "components"},
        base_resolution=(58, 42),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.6, 2.4), height=(1.0, 1.6), cells_x=(22, 32)),
        coefficient_sampler=_shallow_water_coefficients,
    ),
    "thermal_convection": _options(
        "fluid_thermal_convection",
        TRANSIENT_MIXED_DIRECT,
        dt=0.003,
        t_end=0.012,
        fields={
            "velocity": "components",
            "pressure": "scalar",
            "temperature": "scalar",
        },
        base_resolution=(56, 34),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.6, 2.2), height=(0.8, 1.1), cells_x=(18, 28)),
    ),
    "maxwell_pulse": _options(
        "electromagnetic_maxwell_pulse",
        CONSTANT_LHS_CURL_DIRECT,
        dt=0.003,
        t_end=0.018,
        fields={"electric_field": "components"},
        base_resolution=(56, 38),
        num_frames=6,
        resolution_jitter=(-8, 8),
        domains=_rect(length=(1.2, 1.8), height=(0.8, 1.2), cells_x=(20, 30)),
    ),
}


def random_options_for_pde(name: str) -> RandomPDEOptions | None:
    """Return random-run options for a PDE spec."""
    return _RANDOM_OPTIONS.get(name)
