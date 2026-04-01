"""Shared helpers for vector/scalar divergence-free blocks."""

from dataclasses import dataclass
from typing import Any, Callable

from dolfinx import fem

from plm_data.core.boundary_conditions import apply_vector_dirichlet_bcs
from plm_data.core.config import BoundaryFieldConfig
from plm_data.core.mesh import DomainGeometry


@dataclass
class DivergenceFreePairSystem:
    """Vector/scalar spaces plus BC and periodic metadata for one block pair."""

    V: fem.FunctionSpace
    Q: fem.FunctionSpace
    bcs: list[fem.DirichletBC]
    mpc_primary: Any
    mpc_constraint: Any

    @property
    def mpcs(self) -> list[Any] | None:
        """Return the MPC list expected by blocked dolfinx_mpc problems."""
        if self.mpc_primary is None:
            return None
        return [self.mpc_primary, self.mpc_constraint]

    @property
    def primary_solution_space(self) -> fem.FunctionSpace:
        """Return the active solve space for the vector unknown."""
        if self.mpc_primary is None:
            return self.V
        return self.mpc_primary.function_space

    @property
    def constraint_solution_space(self) -> fem.FunctionSpace:
        """Return the active solve space for the scalar constraint unknown."""
        if self.mpc_constraint is None:
            return self.Q
        return self.mpc_constraint.function_space

    def create_primary_function(self, name: str) -> fem.Function:
        """Allocate a vector function on the active solve space."""
        return fem.Function(self.primary_solution_space, name=name)

    def create_constraint_function(self, name: str) -> fem.Function:
        """Allocate a scalar function on the active solve space."""
        return fem.Function(self.constraint_solution_space, name=name)

    def num_dofs(self) -> int:
        """Return the total solved DOF count across both spaces."""
        primary = self.primary_solution_space
        constraint = self.constraint_solution_space
        return (
            primary.dofmap.index_map.size_global * primary.dofmap.index_map_bs
            + constraint.dofmap.index_map.size_global * constraint.dofmap.index_map_bs
        )


def create_divergence_free_pair(
    create_periodic_constraint: Callable[..., Any],
    domain_geom: DomainGeometry,
    primary_boundary_field: BoundaryFieldConfig,
    parameters: dict[str, float],
    *,
    scalar_degree: int,
    constraint_boundary_field: BoundaryFieldConfig | None = None,
) -> DivergenceFreePairSystem:
    """Create one P(k+1)/P(k) vector-scalar block with BCs and periodic MPCs."""
    msh = domain_geom.mesh
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", scalar_degree + 1, (gdim,)))
    Q = fem.functionspace(msh, ("Lagrange", scalar_degree))

    bcs = apply_vector_dirichlet_bcs(
        V,
        domain_geom,
        primary_boundary_field,
        parameters,
    )
    mpc_primary = create_periodic_constraint(
        V,
        domain_geom,
        primary_boundary_field,
        bcs,
    )

    scalar_boundary_field = (
        primary_boundary_field
        if constraint_boundary_field is None
        else constraint_boundary_field
    )
    mpc_constraint = create_periodic_constraint(
        Q,
        domain_geom,
        scalar_boundary_field,
        [],
    )

    return DivergenceFreePairSystem(
        V=V,
        Q=Q,
        bcs=bcs,
        mpc_primary=mpc_primary,
        mpc_constraint=mpc_constraint,
    )
