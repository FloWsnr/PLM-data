"""Managed linear-problem wrappers and PETSc matrix hooks."""

from collections.abc import Callable, Sequence
from typing import Any

import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    assign,
    set_bc,
)
from petsc4py import PETSc


class ManagedLinearProblem:
    """Wrap a DOLFINx linear problem with optional LHS reuse and hooks."""

    def __init__(
        self,
        problem: Any,
        *,
        reuse_lhs: bool = False,
        reuse_preconditioner: bool = False,
        after_lhs_assembled: Callable[["ManagedLinearProblem"], None] | None = None,
    ):
        self._problem = problem
        self._reuse_lhs = reuse_lhs
        self._reuse_preconditioner = reuse_preconditioner
        self._after_lhs_assembled = after_lhs_assembled
        self._lhs_ready = False
        self._preconditioner_ready = False
        self._pc_reuse_set = False

    @property
    def a(self):
        return self._problem.a

    @property
    def A(self):
        return self._problem.A

    @property
    def b(self):
        return self._problem.b

    @property
    def bcs(self):
        return self._problem.bcs

    @property
    def L(self):
        return self._problem.L

    @property
    def P_mat(self):
        return self._problem.P_mat

    @property
    def preconditioner(self):
        return self._problem.preconditioner

    @property
    def solver(self):
        return self._problem.solver

    @property
    def u(self):
        return self._problem.u

    @property
    def x(self):
        return self._problem.x

    def solve(self):
        if not self._reuse_lhs or not self._lhs_ready:
            self._assemble_lhs()
            self._lhs_ready = True

        self._assemble_rhs()
        self.solver.solve(self.b, self.x)
        dolfinx.la.petsc._ghost_update(
            self.x,
            PETSc.InsertMode.INSERT,
            PETSc.ScatterMode.FORWARD,
        )
        assign(self.x, self.u)
        return self.u

    def _assemble_lhs(self) -> None:
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a, bcs=self.bcs)
        self.A.assemble()

        if self.P_mat is not None:
            if not self._reuse_preconditioner or not self._preconditioner_ready:
                self.P_mat.zeroEntries()
                assemble_matrix(self.P_mat, self.preconditioner, bcs=self.bcs)
                self.P_mat.assemble()
                self._preconditioner_ready = True

        if self._after_lhs_assembled is not None:
            self._after_lhs_assembled(self)

        if (
            self._reuse_preconditioner
            and self._preconditioner_ready
            and not self._pc_reuse_set
        ):
            self.solver.getPC().setReusePreconditioner(True)
            self._pc_reuse_set = True

    def _assemble_rhs(self) -> None:
        dolfinx.la.petsc._zero_vector(self.b)
        assemble_vector(self.b, self.L)

        if self.bcs:
            if isinstance(self.u, Sequence):
                bcs1 = fem.bcs_by_block(
                    fem.extract_function_spaces(self.a, 1), self.bcs
                )
                apply_lifting(self.b, self.a, bcs=bcs1)
                dolfinx.la.petsc._ghost_update(
                    self.b,
                    PETSc.InsertMode.ADD,
                    PETSc.ScatterMode.REVERSE,
                )
                bcs0 = fem.bcs_by_block(fem.extract_function_spaces(self.L), self.bcs)
                set_bc(self.b, bcs0)
            else:
                apply_lifting(self.b, [self.a], bcs=[self.bcs])
                dolfinx.la.petsc._ghost_update(
                    self.b,
                    PETSc.InsertMode.ADD,
                    PETSc.ScatterMode.REVERSE,
                )
                for bc in self.bcs:
                    bc.set(self.b.array_w)
        else:
            dolfinx.la.petsc._ghost_update(
                self.b,
                PETSc.InsertMode.ADD,
                PETSc.ScatterMode.REVERSE,
            )


def create_pressure_nullspace(
    problem: ManagedLinearProblem,
    *,
    pressure_block: int,
) -> PETSc.NullSpace:
    """Create a constant-pressure nullspace for a nested mixed system."""
    template = problem.b if problem.b.getType() == PETSc.Vec.Type.NEST else problem.x
    null_vec = template.duplicate()
    sub_vectors = null_vec.getNestSubVecs()
    for vec in sub_vectors:
        vec.set(0.0)
    sub_vectors[pressure_block].set(1.0)
    null_vec.normalize()
    return PETSc.NullSpace().create(vectors=[null_vec])


def configure_nested_problem(
    problem: ManagedLinearProblem,
    *,
    pressure_block: int,
    system_spd_blocks: tuple[int, ...] = (),
    preconditioner_spd_blocks: tuple[int, ...] = (),
) -> None:
    """Attach nullspace and SPD hints to a nested mixed linear system."""
    if problem.A.getType() != PETSc.Mat.Type.NEST:
        return

    problem.A.setNullSpace(
        create_pressure_nullspace(problem, pressure_block=pressure_block)
    )
    for block in system_spd_blocks:
        problem.A.getNestSubMatrix(block, block).setOption(PETSc.Mat.Option.SPD, True)

    if problem.P_mat is None or problem.P_mat.getType() != PETSc.Mat.Type.NEST:
        return

    for block in preconditioner_spd_blocks:
        problem.P_mat.getNestSubMatrix(block, block).setOption(
            PETSc.Mat.Option.SPD, True
        )
