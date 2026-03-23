"""General-purpose FEM utility functions for DOLFINx."""

import ufl
from dolfinx import default_real_type, fem
from dolfinx.mesh import Mesh
from mpi4py import MPI


def domain_average(msh: Mesh, v) -> float:
    """Compute the average of a function over the domain."""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(
            fem.form(fem.Constant(msh, default_real_type(1.0)) * ufl.dx)  # type: ignore[reportArgumentType]
        ),
        op=MPI.SUM,
    )
    return (1 / vol) * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * ufl.dx)),  # type: ignore[reportArgumentType]
        op=MPI.SUM,
    )


def dg_jump(phi, n):
    """Compute the tensor jump of phi across facets: phi+ x n+ + phi- x n-."""
    return ufl.outer(phi("+"), n("+")) + ufl.outer(phi("-"), n("-"))
