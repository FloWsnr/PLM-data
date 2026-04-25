"""Keller-Segel chemotaxis PDE."""

import ufl
from dolfinx import fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.fields import (
    build_ufl_field,
    resolve_param_ref,
    scalar_expression_to_config,
)
from plm_data.stochastic import build_scalar_state_stochastic_term
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import (
    validate_boundary_field_structure,
    validate_scalar_standard_boundary_field,
)
from plm_data.pdes.metadata import (
    PDESpec,
)
from plm_data.pdes.keller_segel.spec import PDE_SPEC


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _chemotaxis_boundary_flux_form(
    chi_rho_n,
    c_n: fem.Function,
    rho_test,
    domain_geom,
    c_boundary_field,
    parameters: dict[str, float],
):
    msh = domain_geom.mesh
    boundary_form = None

    for name, entries in c_boundary_field.sides.items():
        tag = domain_geom.boundary_names[name]
        for bc in entries:
            if bc.type == "periodic":
                continue
            if bc.type == "dirichlet":
                raise ValueError(
                    "Keller-Segel chemoattractant field 'c' does not support "
                    "Dirichlet boundaries; use neumann, robin, or periodic."
                )
            if bc.value is None:
                raise ValueError(
                    f"Boundary '{name}' operator '{bc.type}' requires a value"
                )

            field_config = scalar_expression_to_config(bc.value)
            if field_config["type"] == "custom":
                raise ValueError(
                    f"Boundary '{name}' cannot use custom scalar values for 'c'"
                )

            boundary_flux = build_ufl_field(msh, field_config, parameters)
            if bc.type == "robin":
                alpha = resolve_param_ref(bc.operator_parameters["alpha"], parameters)
                boundary_flux = boundary_flux - alpha * c_n

            term = -ufl.inner(chi_rho_n * boundary_flux, rho_test) * domain_geom.ds(tag)
            boundary_form = term if boundary_form is None else boundary_form + term

    return boundary_form


class _KellerSegelProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        rho_boundary_field = self.config.boundary_field("rho")
        c_boundary_field = self.config.boundary_field("c")

        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="rho",
            boundary_field=rho_boundary_field,
            domain_geom=domain_geom,
        )
        validate_boundary_field_structure(
            pde_name=self.spec.name,
            field_name="c",
            boundary_field=c_boundary_field,
            domain_geom=domain_geom,
            allowed_operators={"neumann", "robin", "periodic"},
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        rho_input = self.config.input("rho")
        c_input = self.config.input("c")
        rho_boundary_field = self.config.boundary_field("rho")
        c_boundary_field = self.config.boundary_field("c")

        assert self.config.time is not None
        dt = self.config.time.dt
        D_rho = self.config.parameters["D_rho"]
        D_c = self.config.parameters["D_c"]
        chi0 = self.config.parameters["chi0"]
        alpha = self.config.parameters["alpha"]
        beta = self.config.parameters["beta"]
        r = self.config.parameters["r"]
        if min(D_rho, D_c, chi0, alpha, beta) <= 0.0 or r < 0.0:
            raise ValueError(
                "Keller-Segel diffusion, chemotaxis, production, and degradation "
                "parameters must be positive; growth rate r cannot be negative."
            )

        rho_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            rho_boundary_field,
            self.config.parameters,
        )
        c_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            c_boundary_field,
            self.config.parameters,
        )
        rho_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            rho_boundary_field,
            rho_bcs,
        )
        c_mpc = self.create_periodic_constraint(V, domain_geom, c_boundary_field, c_bcs)

        self.V_rho = V if rho_mpc is None else rho_mpc.function_space
        self.V_c = V if c_mpc is None else c_mpc.function_space
        self._num_dofs = _space_num_dofs(self.V_rho) + _space_num_dofs(self.V_c)

        self.rho_n = fem.Function(self.V_rho, name="rho")
        self.c_n = fem.Function(self.V_c, name="c")
        self.rho_h = fem.Function(self.V_rho, name="rho_next")
        self.c_h = fem.Function(self.V_c, name="c_next")

        assert rho_input.initial_condition is not None
        apply_ic(
            self.rho_n,
            rho_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        if rho_mpc is not None:
            rho_mpc.backsubstitution(self.rho_n)
        assert c_input.initial_condition is not None
        apply_ic(
            self.c_n,
            c_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
        )
        if c_mpc is not None:
            c_mpc.backsubstitution(self.c_n)
        self.rho_n.x.scatter_forward()
        self.c_n.x.scatter_forward()

        rho_trial = ufl.TrialFunction(V)
        rho_test = ufl.TestFunction(V)
        c_trial = ufl.TrialFunction(V)
        c_test = ufl.TestFunction(V)

        chi_rho_n = chi0 * self.rho_n / (1 + self.rho_n**2)
        growth = r * self.rho_n * (1 - self.rho_n)

        lhs_rho = (
            ufl.inner(rho_trial, rho_test) * ufl.dx
            + dt * D_rho * ufl.inner(ufl.grad(rho_trial), ufl.grad(rho_test)) * ufl.dx
        )
        rhs_rho = (
            ufl.inner(self.rho_n, rho_test) * ufl.dx
            + dt
            * ufl.inner(chi_rho_n * ufl.grad(self.c_n), ufl.grad(rho_test))
            * ufl.dx
            + dt * ufl.inner(growth, rho_test) * ufl.dx
        )
        c_boundary_flux = _chemotaxis_boundary_flux_form(
            chi_rho_n,
            self.c_n,
            rho_test,
            domain_geom,
            c_boundary_field,
            self.config.parameters,
        )
        if c_boundary_flux is not None:
            rhs_rho = rhs_rho + dt * c_boundary_flux

        lhs_c = (
            ufl.inner(c_trial, c_test) * ufl.dx
            + dt * D_c * ufl.inner(ufl.grad(c_trial), ufl.grad(c_test)) * ufl.dx
            + dt * beta * ufl.inner(c_trial, c_test) * ufl.dx
        )
        rhs_c = (
            ufl.inner(self.c_n, c_test) * ufl.dx
            + dt * alpha * ufl.inner(self.rho_n, c_test) * ufl.dx
        )

        self._dynamic_noise_runtimes = []
        rho_stochastic_term, rho_stochastic_runtime = (
            build_scalar_state_stochastic_term(
                self,
                state_name="rho",
                previous_state=self.rho_n,
                test=rho_test,
                dt=dt,
            )
        )
        if rho_stochastic_term is not None and rho_stochastic_runtime is not None:
            rhs_rho = rhs_rho + rho_stochastic_term
            self._dynamic_noise_runtimes.append(rho_stochastic_runtime)

        c_stochastic_term, c_stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="c",
            previous_state=self.c_n,
            test=c_test,
            dt=dt,
        )
        if c_stochastic_term is not None and c_stochastic_runtime is not None:
            rhs_c = rhs_c + c_stochastic_term
            self._dynamic_noise_runtimes.append(c_stochastic_runtime)

        lhs_rho_bc, rhs_rho_bc = build_natural_bc_forms(
            rho_trial,
            rho_test,
            domain_geom,
            rho_boundary_field,
            self.config.parameters,
        )
        if lhs_rho_bc is not None:
            lhs_rho = lhs_rho + dt * lhs_rho_bc
        if rhs_rho_bc is not None:
            rhs_rho = rhs_rho + dt * rhs_rho_bc

        lhs_c_bc, rhs_c_bc = build_natural_bc_forms(
            c_trial,
            c_test,
            domain_geom,
            c_boundary_field,
            self.config.parameters,
        )
        if lhs_c_bc is not None:
            lhs_c = lhs_c + dt * lhs_c_bc
        if rhs_c_bc is not None:
            rhs_c = rhs_c + dt * rhs_c_bc

        self._rho_problem = self.create_linear_problem(
            lhs_rho,
            rhs_rho,
            u=self.rho_h,
            bcs=rho_bcs,
            petsc_options_prefix="plm_keller_segel_rho_",
            mpc=rho_mpc,
        )
        self._c_problem = self.create_linear_problem(
            lhs_c,
            rhs_c,
            u=self.c_h,
            bcs=c_bcs,
            petsc_options_prefix="plm_keller_segel_c_",
            mpc=c_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._rho_problem.solve()
        self._c_problem.solve()

        self.rho_n.x.array[:] = self.rho_h.x.array
        self.rho_n.x.scatter_forward()
        self.c_n.x.array[:] = self.c_h.x.array
        self.c_n.x.scatter_forward()

        return (
            self._rho_problem.solver.getConvergedReason() > 0
            and self._c_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"rho": self.rho_n, "c": self.c_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


class KellerSegelPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _KellerSegelProblem(self.spec, config)
