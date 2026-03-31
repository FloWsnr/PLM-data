"""Kuramoto-Sivashinsky equation preset."""

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import NONLINEAR_MIXED_DIRECT
from plm_data.presets import register_preset
from plm_data.presets.base import (
    PDEPreset,
    ProblemInstance,
    TransientNonlinearProblem,
)
from plm_data.presets.boundary_validation import validate_boundary_field_structure
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_KS_SPEC = PresetSpec(
    name="kuramoto_sivashinsky",
    category="physics",
    description=(
        "Kuramoto-Sivashinsky equation for spatiotemporal chaos using a mixed "
        "u/v formulation where v = -laplacian(u)."
    ),
    equations={
        "u": "du/dt = laplacian(v) + v - (1/2)|grad(u)|^2",
        "v": "v + laplacian(u) = 0",
    },
    parameters=[
        PDEParameter("theta", "Time-stepping parameter (0.5 = Crank-Nicolson)"),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        )
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the primary field u.",
        )
    },
    states={
        "u": StateSpec(name="u", shape="scalar"),
        "v": StateSpec(name="v", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
        "v": OutputSpec(
            name="v",
            shape="scalar",
            output_mode="scalar",
            source_name="v",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _KuramotoSivashinskyProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        validate_boundary_field_structure(
            preset_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
            allowed_operators={"periodic"},
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        boundary_field = self.config.boundary_field("u")

        P1 = element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        ME = fem.functionspace(self.msh, mixed_element([P1, P1]))

        theta = self.config.parameters["theta"]
        dt = self.config.time.dt

        mpc = self.create_periodic_constraint(
            ME,
            domain_geom,
            boundary_field,
            [],
            constrained_spaces=[ME.sub(0), ME.sub(1)],
        )
        solution_space = ME if mpc is None else mpc.function_space

        self.u = fem.Function(solution_space)
        self.u0 = fem.Function(solution_space)

        initial_condition = self.config.input("u").initial_condition
        assert initial_condition is not None
        apply_ic(
            self.u.sub(0),
            initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.u.x.scatter_forward()
        self.u0.x.array[:] = self.u.x.array
        self.u0.x.scatter_forward()

        u, v = ufl.split(self.u)
        u0, v0 = ufl.split(self.u0)
        q, w = ufl.TestFunctions(ME)

        v_theta = (1.0 - theta) * v0 + theta * v
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        # Evolution equation: du/dt - laplacian(v) - v + (1/2)|grad(u)|^2 = 0
        F0 = (
            ufl.inner(u, q) * ufl.dx
            - ufl.inner(u0, q) * ufl.dx
            + dt_c * ufl.inner(ufl.grad(v_theta), ufl.grad(q)) * ufl.dx
            - dt_c * ufl.inner(v_theta, q) * ufl.dx
            + dt_c * 0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) * q * ufl.dx
        )
        # Constraint equation: v + laplacian(u) = 0
        F1 = ufl.inner(v, w) * ufl.dx - ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
        F = F0 + F1
        J = ufl.derivative(F, self.u, ufl.TrialFunction(ME))

        self.problem = self.create_nonlinear_problem(
            F,
            self.u,
            bcs=[],
            petsc_options_prefix="plm_ks_",
            J=J,
            mpc=mpc,
        )

        V0, self._u_dofs = self.u.function_space.sub(0).collapse()
        self.u_out = fem.Function(V0, name="u")
        V1, self._v_dofs = self.u.function_space.sub(1).collapse()
        self.v_out = fem.Function(V1, name="v")

    def step(self, t: float, dt: float) -> bool:
        self.u.x.scatter_forward()
        self.u0.x.array[:] = self.u.x.array
        self.u0.x.scatter_forward()
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason() > 0
        if converged:
            self.u.x.scatter_forward()
        return converged

    def get_output_fields(self) -> dict[str, fem.Function]:
        self.u.x.scatter_forward()
        self.u_out.x.array[:] = self.u.x.array[self._u_dofs]
        self.v_out.x.array[:] = self.u.x.array[self._v_dofs]
        self.u_out.x.scatter_forward()
        self.v_out.x.scatter_forward()
        return {"u": self.u_out, "v": self.v_out}

    def get_num_dofs(self) -> int:
        return self.u.function_space.dofmap.index_map.size_global


@register_preset("kuramoto_sivashinsky")
class KuramotoSivashinskyPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _KS_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _KuramotoSivashinskyProblem(self.spec, config)
