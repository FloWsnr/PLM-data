"""Superlattice reaction-diffusion PDE."""

from dataclasses import dataclass

import ufl
from dolfinx import fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.core.stochastic import build_scalar_state_stochastic_term
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import validate_scalar_standard_boundary_field
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_FIELD_NAMES = ("u_1", "v_1", "u_2", "v_2")

_SUPERLATTICE_SPEC = PDESpec(
    name="superlattice",
    category="physics",
    description=(
        "Coupled Brusselator and Lengyel-Epstein reaction-diffusion systems for "
        "superlattice pattern formation."
    ),
    equations={
        "u_1": (
            "du_1/dt = D_u1 * laplacian(u_1) + a - (b + 1) * u_1 "
            "+ u_1^2 * v_1 + alpha * u_1 * u_2 * (u_2 - u_1)"
        ),
        "v_1": "dv_1/dt = D_v1 * laplacian(v_1) + b * u_1 - u_1^2 * v_1",
        "u_2": (
            "du_2/dt = D_u2 * laplacian(u_2) + c - u_2 "
            "- 4 * u_2 * v_2 / (1 + u_2^2) "
            "+ alpha * u_1 * u_2 * (u_1 - u_2)"
        ),
        "v_2": "dv_2/dt = D_v2 * laplacian(v_2) + d * (u_2 - u_2 * v_2 / (1 + u_2^2))",
    },
    parameters=[
        PDEParameter("D_u1", "Diffusion coefficient of Brusselator activator u_1"),
        PDEParameter("D_v1", "Diffusion coefficient of Brusselator inhibitor v_1"),
        PDEParameter("D_u2", "Diffusion coefficient of Lengyel-Epstein activator u_2"),
        PDEParameter("D_v2", "Diffusion coefficient of Lengyel-Epstein inhibitor v_2"),
        PDEParameter("a", "Brusselator kinetic parameter a"),
        PDEParameter("b", "Brusselator kinetic parameter b"),
        PDEParameter("c", "Lengyel-Epstein feed parameter c"),
        PDEParameter("d", "Lengyel-Epstein kinetic parameter d"),
        PDEParameter("alpha", "Nonlinear coupling strength"),
    ],
    inputs={
        field_name: InputSpec(
            name=field_name,
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        )
        for field_name in _FIELD_NAMES
    },
    boundary_fields={
        field_name: BoundaryFieldSpec(
            name=field_name,
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description=f"Boundary conditions for {field_name}.",
        )
        for field_name in _FIELD_NAMES
    },
    states={
        field_name: StateSpec(
            name=field_name,
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        )
        for field_name in _FIELD_NAMES
    },
    outputs={
        field_name: OutputSpec(
            name=field_name,
            shape="scalar",
            output_mode="scalar",
            source_name=field_name,
        )
        for field_name in _FIELD_NAMES
    },
    static_fields=[],
    supported_dimensions=[2],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


@dataclass
class _ScalarFieldState:
    name: str
    V: fem.FunctionSpace
    current: fem.Function
    next: fem.Function
    bcs: list[fem.DirichletBC]
    mpc: object | None


class _SuperlatticeProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        for field_name in _FIELD_NAMES:
            validate_scalar_standard_boundary_field(
                pde_name=self.spec.name,
                field_name=field_name,
                boundary_field=self.config.boundary_field(field_name),
                domain_geom=domain_geom,
            )

    def _create_field_state(
        self,
        *,
        name: str,
        V: fem.FunctionSpace,
        domain_geom,
        seed_offset: int,
    ) -> _ScalarFieldState:
        boundary_field = self.config.boundary_field(name)
        bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            boundary_field,
            bcs,
        )
        solution_space = V if mpc is None else mpc.function_space

        current = fem.Function(solution_space, name=name)
        next_state = fem.Function(solution_space, name=f"{name}_next")

        initial_condition = self.config.input(name).initial_condition
        assert initial_condition is not None
        seed = None if self.config.seed is None else self.config.seed + seed_offset
        apply_ic(
            current,
            initial_condition,
            self.config.parameters,
            seed=seed,
        )
        if mpc is not None:
            mpc.backsubstitution(current)
        current.x.scatter_forward()

        return _ScalarFieldState(
            name=name,
            V=solution_space,
            current=current,
            next=next_state,
            bcs=bcs,
            mpc=mpc,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))
        params = self.config.parameters
        assert self.config.time is not None
        dt = self.config.time.dt
        if (
            min(
                params["D_u1"],
                params["D_v1"],
                params["D_u2"],
                params["D_v2"],
                params["a"],
                params["b"],
                params["c"],
                params["d"],
            )
            <= 0.0
            or params["alpha"] < 0.0
        ):
            raise ValueError(
                "Superlattice diffusion and kinetic parameters must be positive; "
                "coupling alpha cannot be negative."
            )

        self._fields = {
            field_name: self._create_field_state(
                name=field_name,
                V=V,
                domain_geom=domain_geom,
                seed_offset=index,
            )
            for index, field_name in enumerate(_FIELD_NAMES)
        }
        self._num_dofs = sum(
            _space_num_dofs(field_state.V) for field_state in self._fields.values()
        )

        u_1 = self._fields["u_1"].current
        v_1 = self._fields["v_1"].current
        u_2 = self._fields["u_2"].current
        v_2 = self._fields["v_2"].current

        u_1_trial = ufl.TrialFunction(V)
        u_1_test = ufl.TestFunction(V)
        v_1_trial = ufl.TrialFunction(V)
        v_1_test = ufl.TestFunction(V)
        u_2_trial = ufl.TrialFunction(V)
        u_2_test = ufl.TestFunction(V)
        v_2_trial = ufl.TrialFunction(V)
        v_2_test = ufl.TestFunction(V)

        brusselator_cubic = u_1**2 * v_1
        le_damping = (4.0 * u_2 * v_2) / (1.0 + u_2**2)
        coupling_u_1 = params["alpha"] * u_1 * u_2 * (u_2 - u_1)
        coupling_u_2 = params["alpha"] * u_1 * u_2 * (u_1 - u_2)
        le_v_rhs = params["d"] * (u_2 - (u_2 * v_2) / (1.0 + u_2**2))

        lhs_u_1 = (
            ufl.inner(u_1_trial, u_1_test) * ufl.dx
            + dt
            * params["D_u1"]
            * ufl.inner(ufl.grad(u_1_trial), ufl.grad(u_1_test))
            * ufl.dx
            + dt * (params["b"] + 1.0) * ufl.inner(u_1_trial, u_1_test) * ufl.dx
        )
        rhs_u_1 = (
            ufl.inner(u_1, u_1_test) * ufl.dx
            + dt
            * ufl.inner(
                params["a"] + brusselator_cubic + coupling_u_1,
                u_1_test,
            )
            * ufl.dx
        )

        lhs_v_1 = (
            ufl.inner(v_1_trial, v_1_test) * ufl.dx
            + dt
            * params["D_v1"]
            * ufl.inner(ufl.grad(v_1_trial), ufl.grad(v_1_test))
            * ufl.dx
        )
        rhs_v_1 = (
            ufl.inner(v_1, v_1_test) * ufl.dx
            + dt * ufl.inner(params["b"] * u_1 - brusselator_cubic, v_1_test) * ufl.dx
        )

        lhs_u_2 = (
            ufl.inner(u_2_trial, u_2_test) * ufl.dx
            + dt
            * params["D_u2"]
            * ufl.inner(ufl.grad(u_2_trial), ufl.grad(u_2_test))
            * ufl.dx
            + dt * ufl.inner(u_2_trial, u_2_test) * ufl.dx
        )
        rhs_u_2 = (
            ufl.inner(u_2, u_2_test) * ufl.dx
            + dt
            * ufl.inner(
                params["c"] - le_damping + coupling_u_2,
                u_2_test,
            )
            * ufl.dx
        )

        lhs_v_2 = (
            ufl.inner(v_2_trial, v_2_test) * ufl.dx
            + dt
            * params["D_v2"]
            * ufl.inner(ufl.grad(v_2_trial), ufl.grad(v_2_test))
            * ufl.dx
        )
        rhs_v_2 = (
            ufl.inner(v_2, v_2_test) * ufl.dx
            + dt * ufl.inner(le_v_rhs, v_2_test) * ufl.dx
        )

        forms = {
            "u_1": (u_1_trial, u_1_test, lhs_u_1, rhs_u_1, "plm_superlattice_u_1_"),
            "v_1": (v_1_trial, v_1_test, lhs_v_1, rhs_v_1, "plm_superlattice_v_1_"),
            "u_2": (u_2_trial, u_2_test, lhs_u_2, rhs_u_2, "plm_superlattice_u_2_"),
            "v_2": (v_2_trial, v_2_test, lhs_v_2, rhs_v_2, "plm_superlattice_v_2_"),
        }

        self._problems = {}
        self._dynamic_noise_runtimes = []
        for field_name, (trial, test, lhs_form, rhs_form, prefix) in forms.items():
            boundary_field = self.config.boundary_field(field_name)
            lhs_bc, rhs_bc = build_natural_bc_forms(
                trial,
                test,
                domain_geom,
                boundary_field,
                self.config.parameters,
            )
            if lhs_bc is not None:
                lhs_form = lhs_form + dt * lhs_bc
            if rhs_bc is not None:
                rhs_form = rhs_form + dt * rhs_bc
            stochastic_term, runtime = build_scalar_state_stochastic_term(
                self,
                state_name=field_name,
                previous_state=self._fields[field_name].current,
                test=test,
                dt=dt,
            )
            if stochastic_term is not None and runtime is not None:
                rhs_form = rhs_form + stochastic_term
                self._dynamic_noise_runtimes.append(runtime)

            field_state = self._fields[field_name]
            self._problems[field_name] = self.create_linear_problem(
                lhs_form,
                rhs_form,
                u=field_state.next,
                bcs=field_state.bcs,
                petsc_options_prefix=prefix,
                mpc=field_state.mpc,
            )

    def step(self, t: float, dt: float) -> bool:
        converged = True
        for field_name in _FIELD_NAMES:
            problem = self._problems[field_name]
            problem.solve()
            converged = converged and problem.solver.getConvergedReason() > 0

        for field_name in _FIELD_NAMES:
            field_state = self._fields[field_name]
            field_state.current.x.array[:] = field_state.next.x.array
            field_state.current.x.scatter_forward()

        return converged

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {
            field_name: field_state.current
            for field_name, field_state in self._fields.items()
        }

    def get_num_dofs(self) -> int:
        return self._num_dofs


class SuperlatticePDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _SUPERLATTICE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _SuperlatticeProblem(self.spec, config)
