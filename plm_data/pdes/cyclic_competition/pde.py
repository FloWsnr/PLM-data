"""Cyclic competition reaction-diffusion PDE."""

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
from plm_data.pdes.cyclic_competition.spec import PDE_SPEC


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _CyclicCompetitionProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        for field_name in ("u", "v", "w"):
            validate_scalar_standard_boundary_field(
                pde_name=self.spec.name,
                field_name=field_name,
                boundary_field=self.config.boundary_field(field_name),
                domain_geom=domain_geom,
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        assert self.config.time is not None
        dt = self.config.time.dt
        params = self.config.parameters
        Du = params["Du"]
        Dv = params["Dv"]
        Dw = params["Dw"]
        a_param = params["a"]
        b_param = params["b"]
        if min(Du, Dv, Dw, a_param) <= 0.0 or b_param <= 1.0:
            raise ValueError(
                "Cyclic competition requires positive diffusion coefficients, "
                "a > 0, and b > 1."
            )

        u_boundary_field = self.config.boundary_field("u")
        v_boundary_field = self.config.boundary_field("v")
        w_boundary_field = self.config.boundary_field("w")

        u_bcs = apply_dirichlet_bcs(V, domain_geom, u_boundary_field, params)
        v_bcs = apply_dirichlet_bcs(V, domain_geom, v_boundary_field, params)
        w_bcs = apply_dirichlet_bcs(V, domain_geom, w_boundary_field, params)

        u_mpc = self.create_periodic_constraint(V, domain_geom, u_boundary_field, u_bcs)
        v_mpc = self.create_periodic_constraint(V, domain_geom, v_boundary_field, v_bcs)
        w_mpc = self.create_periodic_constraint(V, domain_geom, w_boundary_field, w_bcs)

        self.V_u = V if u_mpc is None else u_mpc.function_space
        self.V_v = V if v_mpc is None else v_mpc.function_space
        self.V_w = V if w_mpc is None else w_mpc.function_space
        self._num_dofs = (
            _space_num_dofs(self.V_u)
            + _space_num_dofs(self.V_v)
            + _space_num_dofs(self.V_w)
        )

        self.u_n = fem.Function(self.V_u, name="u")
        self.v_n = fem.Function(self.V_v, name="v")
        self.w_n = fem.Function(self.V_w, name="w")
        self.u_h = fem.Function(self.V_u, name="u_next")
        self.v_h = fem.Function(self.V_v, name="v_next")
        self.w_h = fem.Function(self.V_w, name="w_next")

        u_ic = self.config.input("u").initial_condition
        v_ic = self.config.input("v").initial_condition
        w_ic = self.config.input("w").initial_condition
        assert u_ic is not None
        assert v_ic is not None
        assert w_ic is not None

        apply_ic(self.u_n, u_ic, params, seed=self.config.seed)
        if u_mpc is not None:
            u_mpc.backsubstitution(self.u_n)
        seed_v = (self.config.seed + 1) if self.config.seed is not None else None
        apply_ic(self.v_n, v_ic, params, seed=seed_v)
        if v_mpc is not None:
            v_mpc.backsubstitution(self.v_n)
        seed_w = (self.config.seed + 2) if self.config.seed is not None else None
        apply_ic(self.w_n, w_ic, params, seed=seed_w)
        if w_mpc is not None:
            w_mpc.backsubstitution(self.w_n)
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()
        self.w_n.x.scatter_forward()

        u_trial = ufl.TrialFunction(V)
        u_test = ufl.TestFunction(V)
        v_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)
        w_trial = ufl.TrialFunction(V)
        w_test = ufl.TestFunction(V)

        reaction_u = self.u_n * (1 - self.u_n - a_param * self.v_n - b_param * self.w_n)
        lhs_u = (
            ufl.inner(u_trial, u_test) * ufl.dx
            + dt * Du * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
        )
        rhs_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt * ufl.inner(reaction_u, u_test) * ufl.dx
        )

        reaction_v = self.v_n * (1 - b_param * self.u_n - self.v_n - a_param * self.w_n)
        lhs_v = (
            ufl.inner(v_trial, v_test) * ufl.dx
            + dt * Dv * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
        )
        rhs_v = (
            ufl.inner(self.v_n, v_test) * ufl.dx
            + dt * ufl.inner(reaction_v, v_test) * ufl.dx
        )

        reaction_w = self.w_n * (1 - a_param * self.u_n - b_param * self.v_n - self.w_n)
        lhs_w = (
            ufl.inner(w_trial, w_test) * ufl.dx
            + dt * Dw * ufl.inner(ufl.grad(w_trial), ufl.grad(w_test)) * ufl.dx
        )
        rhs_w = (
            ufl.inner(self.w_n, w_test) * ufl.dx
            + dt * ufl.inner(reaction_w, w_test) * ufl.dx
        )

        lhs_u_bc, rhs_u_bc = build_natural_bc_forms(
            u_trial,
            u_test,
            domain_geom,
            u_boundary_field,
            params,
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
            params,
        )
        if lhs_v_bc is not None:
            lhs_v = lhs_v + dt * lhs_v_bc
        if rhs_v_bc is not None:
            rhs_v = rhs_v + dt * rhs_v_bc

        lhs_w_bc, rhs_w_bc = build_natural_bc_forms(
            w_trial,
            w_test,
            domain_geom,
            w_boundary_field,
            params,
        )
        if lhs_w_bc is not None:
            lhs_w = lhs_w + dt * lhs_w_bc
        if rhs_w_bc is not None:
            rhs_w = rhs_w + dt * rhs_w_bc

        self._dynamic_noise_runtimes = []
        stochastic_u, runtime_u = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=u_test,
            dt=dt,
        )
        if stochastic_u is not None and runtime_u is not None:
            rhs_u = rhs_u + stochastic_u
            self._dynamic_noise_runtimes.append(runtime_u)
        stochastic_v, runtime_v = build_scalar_state_stochastic_term(
            self,
            state_name="v",
            previous_state=self.v_n,
            test=v_test,
            dt=dt,
        )
        if stochastic_v is not None and runtime_v is not None:
            rhs_v = rhs_v + stochastic_v
            self._dynamic_noise_runtimes.append(runtime_v)
        stochastic_w, runtime_w = build_scalar_state_stochastic_term(
            self,
            state_name="w",
            previous_state=self.w_n,
            test=w_test,
            dt=dt,
        )
        if stochastic_w is not None and runtime_w is not None:
            rhs_w = rhs_w + stochastic_w
            self._dynamic_noise_runtimes.append(runtime_w)

        self._u_problem = self.create_linear_problem(
            lhs_u,
            rhs_u,
            u=self.u_h,
            bcs=u_bcs,
            petsc_options_prefix="plm_cyclic_competition_u_",
            mpc=u_mpc,
        )
        self._v_problem = self.create_linear_problem(
            lhs_v,
            rhs_v,
            u=self.v_h,
            bcs=v_bcs,
            petsc_options_prefix="plm_cyclic_competition_v_",
            mpc=v_mpc,
        )
        self._w_problem = self.create_linear_problem(
            lhs_w,
            rhs_w,
            u=self.w_h,
            bcs=w_bcs,
            petsc_options_prefix="plm_cyclic_competition_w_",
            mpc=w_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._u_problem.solve()
        self._v_problem.solve()
        self._w_problem.solve()

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        self.v_n.x.array[:] = self.v_h.x.array
        self.v_n.x.scatter_forward()
        self.w_n.x.array[:] = self.w_h.x.array
        self.w_n.x.scatter_forward()

        return (
            self._u_problem.solver.getConvergedReason() > 0
            and self._v_problem.solver.getConvergedReason() > 0
            and self._w_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n, "v": self.v_n, "w": self.w_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


class CyclicCompetitionPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _CyclicCompetitionProblem(self.spec, config)
