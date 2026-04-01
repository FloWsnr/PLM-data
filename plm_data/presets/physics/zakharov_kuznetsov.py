"""Zakharov-Kuznetsov equation preset."""

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

_ZK_SPEC = PresetSpec(
    name="zakharov_kuznetsov",
    category="physics",
    description=(
        "Zakharov-Kuznetsov equation for dispersive nonlinear waves using a mixed "
        "u/w formulation where w = -laplacian(u). The 2D/3D generalization of the "
        "Korteweg-de Vries (KdV) equation."
    ),
    equations={
        "u": "du/dt + alpha*u*du/dx - dw/dx = 0",
        "w": "w + laplacian(u) = 0",
    },
    parameters=[
        PDEParameter("alpha", "Nonlinear advection coefficient (6.0 for canonical ZK)"),
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
        "w": StateSpec(name="w", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _ZakharovKuznetsovProblem(TransientNonlinearProblem):
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

        alpha = self.config.parameters["alpha"]
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

        u, w = ufl.split(self.u)
        u0, w0 = ufl.split(self.u0)
        v, q = ufl.TestFunctions(ME)

        w_theta = (1.0 - theta) * w0 + theta * w
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        # Evolution equation:
        #   du/dt + alpha*u*du/dx - dw/dx = 0
        # Conservative form for advection: alpha*u*du/dx = d(alpha*u^2/2)/dx
        # After IBP on both d/dx terms (boundary terms vanish for periodic):
        F0 = (
            ufl.inner(u, v) * ufl.dx
            - ufl.inner(u0, v) * ufl.dx
            - dt_c * (alpha / 2.0) * u**2 * ufl.grad(v)[0] * ufl.dx
            + dt_c * w_theta * ufl.grad(v)[0] * ufl.dx
        )
        # Constraint equation: w + laplacian(u) = 0
        # After IBP on laplacian(u):
        F1 = ufl.inner(w, q) * ufl.dx - ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx
        F = F0 + F1
        J = ufl.derivative(F, self.u, ufl.TrialFunction(ME))

        self.problem = self.create_nonlinear_problem(
            F,
            self.u,
            bcs=[],
            petsc_options_prefix="plm_zk_",
            J=J,
            mpc=mpc,
        )

        V0, self._u_dofs = self.u.function_space.sub(0).collapse()
        self.u_out = fem.Function(V0, name="u")

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
        self.u_out.x.scatter_forward()
        return {"u": self.u_out}

    def get_num_dofs(self) -> int:
        return self.u.function_space.dofmap.index_map.size_global


@register_preset("zakharov_kuznetsov")
class ZakharovKuznetsovPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _ZK_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _ZakharovKuznetsovProblem(self.spec, config)
