"""Shared test fixtures."""

import pytest

from plm_data.core.config import (
    BoundaryConditionConfig,
    DomainConfig,
    FieldConfig,
    FieldExpressionConfig,
    FieldOutputConfig,
    OutputConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)


def constant(value):
    return FieldExpressionConfig(type="constant", params={"value": value})


def scalar_expr(expr_type: str, **params):
    return FieldExpressionConfig(type=expr_type, params=params)


def vector_expr(**components):
    return FieldExpressionConfig(components=components)


@pytest.fixture
def rectangle_domain():
    return DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )


@pytest.fixture
def direct_solver():
    return SolverConfig(options={"ksp_type": "preonly", "pc_type": "lu"})


@pytest.fixture
def heat_config(tmp_path, rectangle_domain, direct_solver):
    return SimulationConfig(
        preset="heat",
        parameters={"kappa": 0.01},
        domain=rectangle_domain,
        fields={
            "u": FieldConfig(
                boundary_conditions={
                    "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                },
                source=scalar_expr("none"),
                initial_condition=scalar_expr(
                    "gaussian_bump",
                    sigma=0.1,
                    amplitude=1.0,
                    center=[0.5, 0.5],
                ),
                output=FieldOutputConfig(mode="scalar"),
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=42,
    )


@pytest.fixture
def cahn_hilliard_config(tmp_path, rectangle_domain):
    return SimulationConfig(
        preset="cahn_hilliard",
        parameters={
            "lmbda": 0.01,
            "barrier_height": 100.0,
            "mobility": 1.0,
            "theta": 0.5,
        },
        domain=rectangle_domain,
        fields={
            "c": FieldConfig(
                initial_condition=scalar_expr(
                    "random_perturbation",
                    mean=0.63,
                    std=0.02,
                ),
                output=FieldOutputConfig(mode="scalar"),
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
        ),
        solver=SolverConfig(
            options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        ),
        time=TimeConfig(dt=5e-6, t_end=5e-6),
        seed=42,
    )


@pytest.fixture
def navier_stokes_config(tmp_path):
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )
    return SimulationConfig(
        preset="navier_stokes",
        parameters={"Re": 25.0, "k": 1.0},
        domain=domain,
        fields={
            "velocity": FieldConfig(
                boundary_conditions={
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
                },
                source=scalar_expr("none"),
                initial_condition=scalar_expr("custom"),
                output=FieldOutputConfig(mode="components"),
            ),
            "pressure": FieldConfig(output=FieldOutputConfig(mode="scalar")),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
        ),
        solver=SolverConfig(
            options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_14": "80",
                "mat_mumps_icntl_24": "1",
                "mat_mumps_icntl_25": "0",
                "ksp_error_if_not_converged": "1",
            }
        ),
        time=TimeConfig(dt=0.1, t_end=0.1),
        seed=42,
    )


@pytest.fixture
def poisson_config(tmp_path, direct_solver):
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )
    return SimulationConfig(
        preset="poisson",
        parameters={"kappa": 1.0, "f_amplitude": 1.0},
        domain=domain,
        fields={
            "u": FieldConfig(
                boundary_conditions={
                    "x-": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                    "x+": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                    "y-": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                    "y+": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                },
                source=scalar_expr(
                    "sine_product",
                    amplitude="param:f_amplitude",
                    kx=1,
                    ky=1,
                ),
                output=FieldOutputConfig(mode="scalar"),
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=1,
            formats=["numpy"],
        ),
        solver=direct_solver,
        seed=42,
    )
