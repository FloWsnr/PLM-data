"""Stokes flow preset using Taylor-Hood (P2/P1) finite elements."""

from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import default_real_type, fem

from plm_data.core.boundary_conditions import build_vector_natural_bc_forms
from plm_data.core.logging import get_logger
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import CustomProblem, PDEPreset, ProblemInstance, RunResult
from plm_data.presets.boundary_validation import (
    validate_vector_standard_boundary_field,
)
from plm_data.presets.fluids._taylor_hood import (
    create_taylor_hood_linear_problem,
    create_taylor_hood_system,
    normalize_pressure,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
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
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the velocity field.",
        )
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
    def validate_boundary_conditions(self, domain_geom):
        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="velocity",
            boundary_field=self.config.boundary_field("velocity"),
            domain_geom=domain_geom,
        )

    def run(self, output: "FrameWriter") -> RunResult:
        logger = get_logger("solver")
        domain_geom = self.load_domain_geometry()
        msh = domain_geom.mesh
        nu = self.config.parameters["nu"]
        velocity_boundary_field = self.config.boundary_field("velocity")
        system = create_taylor_hood_system(
            self.create_periodic_constraint,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
            pressure_degree=1,
            pressure_boundary_field=velocity_boundary_field,
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

        traction_form = build_vector_natural_bc_forms(
            v,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
        )
        if traction_form is not None:
            L_form = L_form + traction_form

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
