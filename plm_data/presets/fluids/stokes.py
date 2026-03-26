"""Stokes flow preset using Taylor-Hood (P2/P1) finite elements."""

from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import default_real_type, fem

from plm_data.core.logging import get_logger
from plm_data.core.mesh import create_domain
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import CustomProblem, PDEPreset, ProblemInstance, RunResult
from plm_data.presets.fluids._taylor_hood import (
    create_taylor_hood_linear_problem,
    create_taylor_hood_system,
    normalize_pressure,
)
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
    def run(self, output: "FrameWriter") -> RunResult:
        logger = get_logger("solver")
        domain_geom = create_domain(self.config.domain)
        msh = domain_geom.mesh
        nu = self.config.parameters["nu"]
        system = create_taylor_hood_system(
            self.create_periodic_constraint,
            domain_geom,
            self.config.input("velocity").boundary_conditions,
            self.config.parameters,
            pressure_degree=1,
            preset_name="Stokes",
        )
        num_dofs = system.num_dofs()
        logger.info("  Solving Stokes problem (%d DOFs)...", num_dofs)

        u, p = ufl.TrialFunctions(system.VQ)
        v, q = ufl.TestFunctions(system.VQ)

        a_form = (
            nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            - ufl.inner(ufl.div(u), q) * ufl.dx
        )

        L_form = (
            ufl.inner(
                fem.Constant(msh, np.zeros(msh.geometry.dim, dtype=default_real_type)),
                v,
            )
            * ufl.dx
            + ufl.inner(fem.Constant(msh, default_real_type(0.0)), q) * ufl.dx
        )

        velocity_field = self.config.input("velocity")
        assert velocity_field.source is not None
        source_form = build_vector_source_form(
            v, msh, velocity_field.source, self.config.parameters
        )
        if source_form is not None:
            L_form = L_form + source_form

        u_h = system.create_velocity_function("velocity")
        p_h = system.create_pressure_function("pressure")
        problem = create_taylor_hood_linear_problem(
            self,
            system,
            a_form,
            L_form,
            velocity=u_h,
            pressure=p_h,
            petsc_options_prefix="plm_stokes_",
        )
        problem.solve()

        reason = problem.solver.getConvergedReason()
        if reason <= 0:
            logger.error("  Solver did not converge (KSP reason=%s)", reason)
            raise RuntimeError(f"Stokes solver did not converge (KSP reason={reason})")

        normalize_pressure(msh, p_h)
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
