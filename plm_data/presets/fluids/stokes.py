"""Stokes flow preset using Taylor-Hood (P2/P1) finite elements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import default_real_type, fem
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.boundary_conditions import apply_vector_dirichlet_bcs
from plm_data.core.fem_utils import domain_average
from plm_data.core.logging import get_logger
from plm_data.core.mesh import create_domain
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import CustomProblem, PDEPreset, ProblemInstance, RunResult
from plm_data.presets.metadata import (
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
)

if TYPE_CHECKING:
    from plm_data.core.output import FrameWriter

_STOKES_SPEC = PresetSpec(
    name="stokes",
    category="fluids",
    description=(
        "Stokes flow using Taylor-Hood (P2/P1) finite elements. "
        "Solves the steady-state incompressible Stokes equations: "
        "-nu * laplacian(u) + grad(p) = f, div(u) = 0."
    ),
    equations={
        "velocity": "-nu * laplacian(u) + grad(p) = f",
        "pressure": "div(u) = 0",
    },
    parameters=[
        PDEParameter("nu", "Kinematic viscosity"),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_boundary_conditions=True,
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
    },
    steady_state=True,
    supported_dimensions=[2, 3],
)


class _StokesProblem(CustomProblem):
    def run(self, output: FrameWriter) -> RunResult:
        logger = get_logger("solver")
        domain_geom = create_domain(self.config.domain)
        msh = domain_geom.mesh
        gdim = msh.geometry.dim

        nu = self.config.parameters["nu"]

        # Taylor-Hood: P2 velocity, P1 pressure
        V = fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
        Q = fem.functionspace(msh, ("Lagrange", 1))
        VQ = ufl.MixedFunctionSpace(V, Q)

        num_dofs = (
            V.dofmap.index_map.size_global * V.dofmap.index_map_bs
            + Q.dofmap.index_map.size_global
        )
        logger.info("  Solving Stokes problem (%d DOFs)...", num_dofs)

        # Trial and test functions from the mixed space
        u, p = ufl.TrialFunctions(VQ)
        v, q = ufl.TestFunctions(VQ)

        # Bilinear form: viscous + pressure coupling (symmetric saddle-point)
        a_form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            - ufl.inner(ufl.div(u), q) * ufl.dx
        )

        # Linear form: body force on velocity + dummy zero on pressure
        L_form = (
            ufl.inner(fem.Constant(msh, np.zeros(gdim, dtype=default_real_type)), v)
            * ufl.dx
            + ufl.inner(fem.Constant(msh, default_real_type(0.0)), q) * ufl.dx
        )

        # Add vector source term if configured
        velocity_field = self.config.input("velocity")
        assert velocity_field.source is not None
        source_form = build_vector_source_form(
            v, msh, velocity_field.source, self.config.parameters
        )
        if source_form is not None:
            L_form = L_form + source_form

        velocity_bcs = velocity_field.boundary_conditions
        for name, bc_config in velocity_bcs.items():
            if bc_config.type != "dirichlet":
                raise ValueError(
                    f"Stokes boundary '{name}' must be dirichlet, got '{bc_config.type}'"
                )
        bcs = apply_vector_dirichlet_bcs(
            V,
            domain_geom,
            velocity_bcs,
            self.config.parameters,
        )

        # Solution functions
        u_h = fem.Function(V, name="velocity")
        p_h = fem.Function(Q, name="pressure")

        # Solve the block system
        problem = LinearProblem(
            ufl.extract_blocks(a_form),
            ufl.extract_blocks(L_form),
            u=[u_h, p_h],
            bcs=bcs,
            kind="mpi",
            petsc_options_prefix="plm_stokes_",
            petsc_options=self._solver_options,
        )
        problem.solve()

        reason = problem.solver.getConvergedReason()
        if reason <= 0:
            logger.error("  Solver did not converge (KSP reason=%s)", reason)
            raise RuntimeError(f"Stokes solver did not converge (KSP reason={reason})")

        # Normalize pressure (remove null-space constant)
        p_h.x.array[:] -= domain_average(msh, p_h)

        output.write_frame({"velocity": u_h, "pressure": p_h}, t=0.0)
        logger.info("  Solve complete (converged)")
        return RunResult(num_dofs=num_dofs, solver_converged=True)


@register_preset("stokes")
class StokesPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _STOKES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _StokesProblem(self.spec, config)
