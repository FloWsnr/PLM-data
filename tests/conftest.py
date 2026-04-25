"""Shared test fixtures."""

import pytest

from plm_data.core.runtime_config import BoundaryConditionConfig
from tests.runtime_helpers import (
    constant,
    make_heat_config,
    rectangle_domain as make_rectangle_domain,
    scalar_expr,
)


@pytest.fixture
def rectangle_domain():
    return make_rectangle_domain()


@pytest.fixture
def heat_config(tmp_path):
    return make_heat_config(
        tmp_path,
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
        coefficients={"kappa": constant(0.01)},
    )
