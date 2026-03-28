"""Shared test fixtures."""

import pytest

from plm_data.core.config import (
    BoundaryConditionConfig,
    DomainConfig,
    InputConfig,
    OutputConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)
from tests.preset_matrix import (
    boundary_field_config,
    constant,
    output_fields,
    scalar_expr,
)


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
        parameters={},
        domain=rectangle_domain,
        inputs={
            "u": InputConfig(
                source=scalar_expr("none"),
                initial_condition=scalar_expr(
                    "gaussian_bump",
                    sigma=0.1,
                    amplitude=1.0,
                    center=[0.5, 0.5],
                ),
            )
        },
        boundary_conditions={
            "u": boundary_field_config(
                {
                    "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                }
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=42,
        coefficients={"kappa": constant(0.01)},
    )
