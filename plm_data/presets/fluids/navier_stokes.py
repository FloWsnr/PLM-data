"""Incompressible Navier-Stokes preset using Taylor-Hood elements."""

import numpy as np
import ufl
from dolfinx import default_real_type, fem
from plm_data.core.boundary_conditions import build_vector_natural_bc_forms
from plm_data.core.initial_conditions import apply_vector_ic
from plm_data.core.linear_problem import configure_nested_problem
from plm_data.core.solver_strategies import (
    TRANSIENT_MIXED_DIRECT,
    TRANSIENT_SADDLE_POINT,
)
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets.fluids._taylor_hood import (
    create_taylor_hood_linear_problem,
    create_taylor_hood_system,
    normalize_pressure,
)
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_vector_standard_boundary_field,
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


_NAVIER_STOKES_SPEC = PresetSpec(
    name="navier_stokes",
    category="fluids",
    description=(
        "Incompressible Navier-Stokes equations using a Taylor-Hood "
        "velocity/pressure discretization."
    ),
    equations={
        "velocity": "du/dt + (u.grad)u = -grad(p) + (1/Re)*laplacian(u)",
        "pressure": "div(u) = 0",
    },
    parameters=[
        PDEParameter("Re", "Reynolds number"),
        PDEParameter("k", "Polynomial degree for velocity and pressure spaces"),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
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
    static_fields=[],
    supported_dimensions=[2, 3],
)


class _NavierStokesProblem(TransientLinearProblem):
    supported_solver_strategies = (
        TRANSIENT_SADDLE_POINT,
        TRANSIENT_MIXED_DIRECT,
    )

    def validate_boundary_conditions(self, domain_geom):
        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="velocity",
            boundary_field=self.config.boundary_field("velocity"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.domain_geom = domain_geom
        self.msh = domain_geom.mesh
        self.gdim = self.msh.geometry.dim

        Re = self.config.parameters["Re"]
        k = int(self.config.parameters["k"])
        dt = self.config.time.dt
        velocity_field = self.config.input("velocity")
        velocity_boundary_field = self.config.boundary_field("velocity")
        system = create_taylor_hood_system(
            self.create_periodic_constraint,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
            pressure_degree=k,
            pressure_boundary_field=velocity_boundary_field,
        )
        self._num_dofs = system.num_dofs()

        u, p = ufl.TrialFunctions(system.VQ)
        v, q = ufl.TestFunctions(system.VQ)
        self.u_h = system.create_velocity_function("velocity")
        self.p_h = system.create_pressure_function("pressure")
        self.u_n = system.create_velocity_function("u_prev")

        delta_t = fem.Constant(self.msh, default_real_type(dt))
        zero_vector = fem.Constant(
            self.msh,
            np.zeros(self.gdim, dtype=default_real_type),
        )
        zero_pressure = fem.Constant(self.msh, default_real_type(0.0))
        base_rhs = (
            ufl.inner(zero_vector, v) * ufl.dx + ufl.inner(zero_pressure, q) * ufl.dx
        )
        assert velocity_field.source is not None
        source_form = build_vector_source_form(
            v,
            self.msh,
            velocity_field.source,
            self.config.parameters,
        )
        if source_form is not None:
            base_rhs = base_rhs + source_form
        traction_form = build_vector_natural_bc_forms(
            v,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
        )
        if traction_form is not None:
            base_rhs = base_rhs + traction_form

        stokes_form = (
            (1.0 / Re) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            - ufl.inner(ufl.div(u), q) * ufl.dx
        )

        assert velocity_field.initial_condition is not None
        if velocity_field.initial_condition.type == "custom":
            stokes_problem = create_taylor_hood_linear_problem(
                self,
                system,
                stokes_form,
                base_rhs,
                velocity=self.u_h,
                pressure=self.p_h,
                petsc_options_prefix="plm_ns_stokes_",
            )
            stokes_problem.solve()
            reason = stokes_problem.solver.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(
                    "Navier-Stokes Stokes initialization did not converge "
                    f"(KSP reason={reason})"
                )
            normalize_pressure(self.msh, self.p_h)
        else:
            apply_vector_ic(
                self.u_h,
                velocity_field.initial_condition,
                self.config.parameters,
                seed=self.config.seed,
            )
            self.u_h.x.scatter_forward()
            self.p_h.x.array[:] = 0.0
            self.p_h.x.scatter_forward()

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()

        a_ns = (
            ufl.inner(u / delta_t, v) * ufl.dx
            + ufl.inner(ufl.dot(ufl.grad(u), self.u_n), v) * ufl.dx
            + (1.0 / Re) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(p, ufl.div(v)) * ufl.dx
            - ufl.inner(ufl.div(u), q) * ufl.dx
        )
        preconditioner_form = [
            [
                ufl.inner(u / delta_t, v) * ufl.dx
                + (1.0 / Re) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
                None,
            ],
            [
                None,
                ufl.inner(p, q) * ufl.dx,
            ],
        ]
        L_ns = ufl.inner(self.u_n / delta_t, v) * ufl.dx + base_rhs

        self._ns_problem = create_taylor_hood_linear_problem(
            self,
            system,
            a_ns,
            L_ns,
            velocity=self.u_h,
            pressure=self.p_h,
            petsc_options_prefix="plm_ns_",
            preconditioner_form=preconditioner_form,
        )

    def step(self, t: float, dt: float) -> bool:
        self._ns_problem.solve()
        normalize_pressure(self.msh, self.p_h)
        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        return self._ns_problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"velocity": self.u_h, "pressure": self.p_h}

    def get_num_dofs(self) -> int:
        return self._num_dofs

    def linear_problem_after_lhs_assembled(self, *, kind=None, mpc=None):
        if kind != "nest" or mpc is not None:
            return None

        def _configure(problem) -> None:
            configure_nested_problem(
                problem,
                pressure_block=1,
                preconditioner_spd_blocks=(0, 1),
            )

        return _configure

    def should_reuse_preconditioner(self, *, mpc=None) -> bool:
        return mpc is None


@register_preset("navier_stokes")
class NavierStokesPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _NAVIER_STOKES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _NavierStokesProblem(self.spec, config)
