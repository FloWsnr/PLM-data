"""FitzHugh-Nagumo reaction-diffusion PDE."""

import ufl
from dolfinx import fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.stochastic import build_scalar_state_stochastic_term
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import validate_scalar_standard_boundary_field
from plm_data.pdes.metadata import (
    PDESpec,
)
from plm_data.pdes.fitzhugh_nagumo.spec import PDE_SPEC


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _FitzHughNagumoProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="v",
            boundary_field=self.config.boundary_field("v"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        u_input = self.config.input("u")
        v_input = self.config.input("v")
        u_boundary_field = self.config.boundary_field("u")
        v_boundary_field = self.config.boundary_field("v")

        assert self.config.time is not None
        dt = self.config.time.dt
        Du = self.config.parameters["Du"]
        Dv = self.config.parameters["Dv"]
        tau = self.config.parameters["tau"]
        b_param = self.config.parameters["b"]
        a_param = self.config.parameters["a"]
        if Du <= 0.0 or Dv <= 0.0:
            raise ValueError("FitzHugh-Nagumo diffusion parameters must be positive.")
        if tau <= 0.0:
            raise ValueError("FitzHugh-Nagumo parameter 'tau' must be positive.")
        if b_param <= 0.0:
            raise ValueError("FitzHugh-Nagumo parameter 'b' must be positive.")

        u_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            u_boundary_field,
            self.config.parameters,
        )
        v_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            v_boundary_field,
            self.config.parameters,
        )
        u_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            u_boundary_field,
            u_bcs,
        )
        v_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            v_boundary_field,
            v_bcs,
        )

        self.V_u = V if u_mpc is None else u_mpc.function_space
        self.V_v = V if v_mpc is None else v_mpc.function_space
        self._num_dofs = _space_num_dofs(self.V_u) + _space_num_dofs(self.V_v)

        self.u_n = fem.Function(self.V_u, name="u")
        self.v_n = fem.Function(self.V_v, name="v")
        self.u_h = fem.Function(self.V_u, name="u_next")
        self.v_h = fem.Function(self.V_v, name="v_next")

        assert u_input.initial_condition is not None
        apply_ic(
            self.u_n,
            u_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        if u_mpc is not None:
            u_mpc.backsubstitution(self.u_n)
        assert v_input.initial_condition is not None
        apply_ic(
            self.v_n,
            v_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
        )
        if v_mpc is not None:
            v_mpc.backsubstitution(self.v_n)
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()

        u_trial = ufl.TrialFunction(V)
        u_test = ufl.TestFunction(V)
        v_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)

        lhs_u = (
            ufl.inner(u_trial, u_test) * ufl.dx
            + dt * Du * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
        )
        rhs_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt * ufl.inner(self.u_n - self.u_n**3 - self.v_n, u_test) * ufl.dx
        )

        lhs_v = (
            tau * ufl.inner(v_trial, v_test) * ufl.dx
            + dt * Dv * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
            + dt * b_param * ufl.inner(v_trial, v_test) * ufl.dx
        )
        rhs_v = (
            tau * ufl.inner(self.v_n, v_test) * ufl.dx
            + dt * ufl.inner(self.u_n + a_param, v_test) * ufl.dx
        )

        self._dynamic_noise_runtimes = []
        u_stochastic_term, u_stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=u_test,
            dt=dt,
        )
        if u_stochastic_term is not None and u_stochastic_runtime is not None:
            rhs_u = rhs_u + u_stochastic_term
            self._dynamic_noise_runtimes.append(u_stochastic_runtime)

        v_stochastic_term, v_stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="v",
            previous_state=self.v_n,
            test=v_test,
            dt=dt,
        )
        if v_stochastic_term is not None and v_stochastic_runtime is not None:
            rhs_v = rhs_v + v_stochastic_term
            self._dynamic_noise_runtimes.append(v_stochastic_runtime)

        lhs_u_bc, rhs_u_bc = build_natural_bc_forms(
            u_trial,
            u_test,
            domain_geom,
            u_boundary_field,
            self.config.parameters,
        )
        if lhs_u_bc is not None:
            lhs_u = lhs_u + dt * lhs_u_bc
        if rhs_u_bc is not None:
            rhs_u = rhs_u + dt * rhs_u_bc

        lhs_v_bc, rhs_v_bc = build_natural_bc_forms(
            v_trial,
            v_test,
            domain_geom,
            v_boundary_field,
            self.config.parameters,
        )
        if lhs_v_bc is not None:
            lhs_v = lhs_v + dt * lhs_v_bc
        if rhs_v_bc is not None:
            rhs_v = rhs_v + dt * rhs_v_bc

        self._u_problem = self.create_linear_problem(
            lhs_u,
            rhs_u,
            u=self.u_h,
            bcs=u_bcs,
            petsc_options_prefix="plm_fitzhugh_nagumo_u_",
            mpc=u_mpc,
        )
        self._v_problem = self.create_linear_problem(
            lhs_v,
            rhs_v,
            u=self.v_h,
            bcs=v_bcs,
            petsc_options_prefix="plm_fitzhugh_nagumo_v_",
            mpc=v_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._u_problem.solve()
        self._v_problem.solve()

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        self.v_n.x.array[:] = self.v_h.x.array
        self.v_n.x.scatter_forward()

        return (
            self._u_problem.solver.getConvergedReason() > 0
            and self._v_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n, "v": self.v_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


class FitzHughNagumoPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _FitzHughNagumoProblem(self.spec, config)
