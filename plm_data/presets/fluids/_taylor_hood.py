"""Shared Taylor-Hood helpers for incompressible flow presets."""

from dataclasses import dataclass
from typing import Any, Callable

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import apply_vector_dirichlet_bcs
from plm_data.core.config import BoundaryConditionConfig
from plm_data.core.fem_utils import domain_average
from plm_data.core.mesh import DomainGeometry


@dataclass
class TaylorHoodSystem:
    """Shared Taylor-Hood spaces, BCs, and optional periodic constraints."""

    V: fem.FunctionSpace
    Q: fem.FunctionSpace
    VQ: Any
    bcs: list[fem.DirichletBC]
    mpc_u: Any
    mpc_p: Any

    @property
    def mpcs(self) -> list[Any] | None:
        """Return the per-block MPC list expected by dolfinx_mpc."""
        if self.mpc_u is None:
            return None
        return [self.mpc_u, self.mpc_p]

    @property
    def velocity_solution_space(self) -> fem.FunctionSpace:
        """Return the unconstrained or MPC-constrained velocity space."""
        if self.mpc_u is None:
            return self.V
        return self.mpc_u.function_space

    @property
    def pressure_solution_space(self) -> fem.FunctionSpace:
        """Return the unconstrained or MPC-constrained pressure space."""
        if self.mpc_p is None:
            return self.Q
        return self.mpc_p.function_space

    def create_velocity_function(self, name: str) -> fem.Function:
        """Allocate a velocity function on the active solve space."""
        return fem.Function(self.velocity_solution_space, name=name)

    def create_pressure_function(self, name: str) -> fem.Function:
        """Allocate a pressure function on the active solve space."""
        return fem.Function(self.pressure_solution_space, name=name)

    def num_dofs(self) -> int:
        """Return the total solved DOF count across velocity and pressure."""
        velocity_space = self.velocity_solution_space
        pressure_space = self.pressure_solution_space
        return (
            velocity_space.dofmap.index_map.size_global
            * velocity_space.dofmap.index_map_bs
            + pressure_space.dofmap.index_map.size_global
            * pressure_space.dofmap.index_map_bs
        )


def create_taylor_hood_system(
    create_periodic_constraint: Callable[..., Any],
    domain_geom: DomainGeometry,
    velocity_bcs: dict[str, BoundaryConditionConfig],
    parameters: dict[str, float],
    *,
    pressure_degree: int,
    preset_name: str,
) -> TaylorHoodSystem:
    """Create shared Taylor-Hood spaces, BCs, and periodic constraints."""
    for name, bc_config in velocity_bcs.items():
        if bc_config.type != "dirichlet":
            raise ValueError(
                f"{preset_name} boundary '{name}' must be dirichlet, got "
                f"'{bc_config.type}'"
            )

    msh = domain_geom.mesh
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", pressure_degree + 1, (gdim,)))
    Q = fem.functionspace(msh, ("Lagrange", pressure_degree))
    VQ = ufl.MixedFunctionSpace(V, Q)

    bcs = apply_vector_dirichlet_bcs(V, domain_geom, velocity_bcs, parameters)
    mpc_u = create_periodic_constraint(V, domain_geom, bcs)
    mpc_p = create_periodic_constraint(Q, domain_geom, [])

    return TaylorHoodSystem(
        V=V,
        Q=Q,
        VQ=VQ,
        bcs=bcs,
        mpc_u=mpc_u,
        mpc_p=mpc_p,
    )


def create_taylor_hood_linear_problem(
    problem_instance,
    system: TaylorHoodSystem,
    a_form,
    L_form,
    *,
    velocity: fem.Function,
    pressure: fem.Function,
    petsc_options_prefix: str,
):
    """Create a blocked Taylor-Hood linear problem with optional MPCs."""
    return problem_instance.create_linear_problem(
        ufl.extract_blocks(a_form),
        ufl.extract_blocks(L_form),
        u=[velocity, pressure],
        bcs=system.bcs,
        petsc_options_prefix=petsc_options_prefix,
        kind="mpi",
        mpc=system.mpcs,
    )


def normalize_pressure(msh, pressure: fem.Function) -> None:
    """Remove the constant pressure nullspace component."""
    pressure.x.array[:] -= domain_average(msh, pressure)
    pressure.x.scatter_forward()
