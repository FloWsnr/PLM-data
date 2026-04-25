"""Tests for plm_data.boundary_conditions.periodic."""

import pytest

from plm_data.boundary_conditions.periodic import (
    periodic_boundary_names,
    require_dolfinx_mpc,
    require_unverified_periodic_support,
)
from plm_data.core.runtime_config import (
    BoundaryConditionConfig,
    BoundaryFieldConfig,
)


def test_periodic_boundary_names_returns_active_periodic_sides():
    boundary_field = BoundaryFieldConfig(
        sides={
            "x-": [BoundaryConditionConfig(type="periodic", pair_with="x+")],
            "x+": [BoundaryConditionConfig(type="periodic", pair_with="x-")],
            "y-": [BoundaryConditionConfig(type="neumann")],
        }
    )

    assert periodic_boundary_names(boundary_field) == {"x-", "x+"}


def test_require_dolfinx_mpc_raises_clear_error(monkeypatch):
    def _raise_import_error(name):
        assert name == "dolfinx_mpc"
        raise ImportError(name)

    monkeypatch.setattr(
        "plm_data.boundary_conditions.periodic.importlib.import_module",
        _raise_import_error,
    )

    with pytest.raises(RuntimeError, match="Periodic boundary conditions require"):
        require_dolfinx_mpc()


def test_require_unverified_periodic_support_only_rejects_periodic_fields():
    nonperiodic = BoundaryFieldConfig(
        sides={"x-": [BoundaryConditionConfig(type="neumann")]}
    )
    require_unverified_periodic_support("heat", nonperiodic, "scalar spaces")

    periodic = BoundaryFieldConfig(
        sides={"x-": [BoundaryConditionConfig(type="periodic", pair_with="x+")]}
    )
    with pytest.raises(NotImplementedError, match="does not yet verify"):
        require_unverified_periodic_support("heat", periodic, "scalar spaces")
