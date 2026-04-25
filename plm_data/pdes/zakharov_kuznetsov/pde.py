"""Zakharov-Kuznetsov equation PDE."""

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem

from plm_data.boundary_conditions.runtime import apply_dirichlet_bcs_to_subspace
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.runtime_config import BoundaryFieldConfig
from plm_data.core.solver_strategies import NONLINEAR_MIXED_DIRECT
from plm_data.fields import is_exact_zero_field_expression
from plm_data.pdes.base import PDE, ProblemInstance, TransientNonlinearProblem
from plm_data.pdes.boundary_validation import validate_boundary_field_structure
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_ZK_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name]
    for name in ("dirichlet", "neumann", "periodic")
}

_ZK_SPEC = PDESpec(
    name="zakharov_kuznetsov",
    category="physics",
    description=(
        "Zakharov-Kuznetsov equation for dispersive nonlinear waves using a "
        "mixed u/w formulation where w = -laplacian(u)."
    ),
    equations={
        "u": "du/dt + alpha*u*du/dx - dw/dx = 0",
        "w": "w + laplacian(u) = 0",
    },
    parameters=[
        PDEParameter("alpha", "Nonlinear advection coefficient"),
        PDEParameter("theta", "Time-stepping parameter"),
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
            operators=_ZK_BOUNDARY_OPERATORS,
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
    supported_dimensions=[2],
)


def _validate_homogeneous_wall_values(
    *,
    boundary_field: BoundaryFieldConfig,
    parameters: dict[str, float],
) -> None:
    for side_name, entries in boundary_field.sides.items():
        for entry in entries:
            if entry.type not in {"dirichlet", "neumann"}:
                continue
            if entry.value is None:
                raise ValueError(
                    "Zakharov-Kuznetsov wall boundary operators require an "
                    f"explicit zero value. Side '{side_name}' is missing one."
                )
            if not is_exact_zero_field_expression(entry.value, parameters):
                raise ValueError(
                    "Zakharov-Kuznetsov currently supports only homogeneous "
                    f"{entry.type} wall values. Side '{side_name}' is nonzero."
                )


def _build_zero_dirichlet_bcs(
    mixed_space: fem.FunctionSpace,
    domain_geom,
    boundary_field: BoundaryFieldConfig,
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    return [
        *apply_dirichlet_bcs_to_subspace(
            mixed_space.sub(0),
            domain_geom,
            boundary_field,
            parameters,
        ),
        *apply_dirichlet_bcs_to_subspace(
            mixed_space.sub(1),
            domain_geom,
            boundary_field,
            parameters,
        ),
    ]


def _is_fully_periodic(boundary_field: BoundaryFieldConfig) -> bool:
    return all(
        entry.type == "periodic"
        for entries in boundary_field.sides.values()
        for entry in entries
    )


class _ZakharovKuznetsovProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        boundary_field = self.config.boundary_field("u")
        validate_boundary_field_structure(
            pde_name=self.spec.name,
            field_name="u",
            boundary_field=boundary_field,
            domain_geom=domain_geom,
            allowed_operators=set(_ZK_BOUNDARY_OPERATORS),
        )
        _validate_homogeneous_wall_values(
            boundary_field=boundary_field,
            parameters=self.config.parameters,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        boundary_field = self.config.boundary_field("u")

        P1 = element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        ME = fem.functionspace(self.msh, mixed_element([P1, P1]))

        assert self.config.time is not None
        alpha = self.config.parameters["alpha"]
        theta = self.config.parameters["theta"]
        dt = self.config.time.dt
        if not 0.0 <= theta <= 1.0:
            raise ValueError("Zakharov-Kuznetsov theta must be between 0 and 1.")

        bcs = _build_zero_dirichlet_bcs(
            ME,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(
            ME,
            domain_geom,
            boundary_field,
            bcs,
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
        if mpc is None:
            for bc in bcs:
                bc.set(self.u.x.array)
        self.u.x.scatter_forward()
        self.u0.x.array[:] = self.u.x.array
        if mpc is None:
            for bc in bcs:
                bc.set(self.u0.x.array)
        self.u0.x.scatter_forward()

        u, w = ufl.split(self.u)
        u0, w0 = ufl.split(self.u0)
        v, q = ufl.TestFunctions(ME)

        w_theta = (1.0 - theta) * w0 + theta * w
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        if _is_fully_periodic(boundary_field):
            F0 = (
                ufl.inner(u, v) * ufl.dx
                - ufl.inner(u0, v) * ufl.dx
                - dt_c * (alpha / 2.0) * u**2 * ufl.grad(v)[0] * ufl.dx
                + dt_c * w_theta * ufl.grad(v)[0] * ufl.dx
            )
        else:
            F0 = (
                ufl.inner(u, v) * ufl.dx
                - ufl.inner(u0, v) * ufl.dx
                + dt_c * alpha * u * ufl.grad(u)[0] * v * ufl.dx
                - dt_c * ufl.grad(w_theta)[0] * v * ufl.dx
            )
        F1 = ufl.inner(w, q) * ufl.dx - ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx
        F = F0 + F1
        J = ufl.derivative(F, self.u, ufl.TrialFunction(ME))

        self.problem = self.create_nonlinear_problem(
            F,
            self.u,
            bcs=bcs,
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


class ZakharovKuznetsovPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _ZK_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _ZakharovKuznetsovProblem(self.spec, config)
