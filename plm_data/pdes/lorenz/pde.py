"""Diffusively coupled Lorenz reaction-diffusion PDE."""

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
from plm_data.pdes.lorenz.spec import PDE_SPEC


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _LorenzProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        for field_name in ("x", "y", "z"):
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

        x_input = self.config.input("x")
        y_input = self.config.input("y")
        z_input = self.config.input("z")
        x_boundary_field = self.config.boundary_field("x")
        y_boundary_field = self.config.boundary_field("y")
        z_boundary_field = self.config.boundary_field("z")

        assert self.config.time is not None
        dt = self.config.time.dt
        sigma = self.config.parameters["sigma"]
        rho = self.config.parameters["rho"]
        beta = self.config.parameters["beta"]
        D = self.config.parameters["D"]
        if min(sigma, rho, beta, D) <= 0.0:
            raise ValueError(
                "Lorenz parameters sigma, rho, beta, and D must be positive."
            )

        x_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            x_boundary_field,
            self.config.parameters,
        )
        y_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            y_boundary_field,
            self.config.parameters,
        )
        z_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            z_boundary_field,
            self.config.parameters,
        )

        x_mpc = self.create_periodic_constraint(V, domain_geom, x_boundary_field, x_bcs)
        y_mpc = self.create_periodic_constraint(V, domain_geom, y_boundary_field, y_bcs)
        z_mpc = self.create_periodic_constraint(V, domain_geom, z_boundary_field, z_bcs)

        self.V_x = V if x_mpc is None else x_mpc.function_space
        self.V_y = V if y_mpc is None else y_mpc.function_space
        self.V_z = V if z_mpc is None else z_mpc.function_space
        self._num_dofs = (
            _space_num_dofs(self.V_x)
            + _space_num_dofs(self.V_y)
            + _space_num_dofs(self.V_z)
        )

        self.x_n = fem.Function(self.V_x, name="x")
        self.y_n = fem.Function(self.V_y, name="y")
        self.z_n = fem.Function(self.V_z, name="z")
        self.x_h = fem.Function(self.V_x, name="x_next")
        self.y_h = fem.Function(self.V_y, name="y_next")
        self.z_h = fem.Function(self.V_z, name="z_next")

        assert x_input.initial_condition is not None
        apply_ic(
            self.x_n,
            x_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        if x_mpc is not None:
            x_mpc.backsubstitution(self.x_n)
        assert y_input.initial_condition is not None
        apply_ic(
            self.y_n,
            y_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
        )
        if y_mpc is not None:
            y_mpc.backsubstitution(self.y_n)
        assert z_input.initial_condition is not None
        apply_ic(
            self.z_n,
            z_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 2) if self.config.seed is not None else None,
        )
        if z_mpc is not None:
            z_mpc.backsubstitution(self.z_n)
        self.x_n.x.scatter_forward()
        self.y_n.x.scatter_forward()
        self.z_n.x.scatter_forward()

        x_trial = ufl.TrialFunction(V)
        x_test = ufl.TestFunction(V)
        y_trial = ufl.TrialFunction(V)
        y_test = ufl.TestFunction(V)
        z_trial = ufl.TrialFunction(V)
        z_test = ufl.TestFunction(V)

        lhs_x = (
            ufl.inner(x_trial, x_test) * ufl.dx
            + dt * D * ufl.inner(ufl.grad(x_trial), ufl.grad(x_test)) * ufl.dx
            + dt * sigma * ufl.inner(x_trial, x_test) * ufl.dx
        )
        rhs_x = (
            ufl.inner(self.x_n, x_test) * ufl.dx
            + dt * sigma * ufl.inner(self.y_n, x_test) * ufl.dx
        )

        lhs_y = (
            ufl.inner(y_trial, y_test) * ufl.dx
            + dt * D * ufl.inner(ufl.grad(y_trial), ufl.grad(y_test)) * ufl.dx
            + dt * ufl.inner(y_trial, y_test) * ufl.dx
        )
        rhs_y = (
            ufl.inner(self.y_n, y_test) * ufl.dx
            + dt * ufl.inner(self.x_n * (rho - self.z_n), y_test) * ufl.dx
        )

        lhs_z = (
            ufl.inner(z_trial, z_test) * ufl.dx
            + dt * D * ufl.inner(ufl.grad(z_trial), ufl.grad(z_test)) * ufl.dx
            + dt * beta * ufl.inner(z_trial, z_test) * ufl.dx
        )
        rhs_z = (
            ufl.inner(self.z_n, z_test) * ufl.dx
            + dt * ufl.inner(self.x_n * self.y_n, z_test) * ufl.dx
        )

        lhs_x_bc, rhs_x_bc = build_natural_bc_forms(
            x_trial,
            x_test,
            domain_geom,
            x_boundary_field,
            self.config.parameters,
        )
        if lhs_x_bc is not None:
            lhs_x = lhs_x + dt * lhs_x_bc
        if rhs_x_bc is not None:
            rhs_x = rhs_x + dt * rhs_x_bc

        lhs_y_bc, rhs_y_bc = build_natural_bc_forms(
            y_trial,
            y_test,
            domain_geom,
            y_boundary_field,
            self.config.parameters,
        )
        if lhs_y_bc is not None:
            lhs_y = lhs_y + dt * lhs_y_bc
        if rhs_y_bc is not None:
            rhs_y = rhs_y + dt * rhs_y_bc

        lhs_z_bc, rhs_z_bc = build_natural_bc_forms(
            z_trial,
            z_test,
            domain_geom,
            z_boundary_field,
            self.config.parameters,
        )
        if lhs_z_bc is not None:
            lhs_z = lhs_z + dt * lhs_z_bc
        if rhs_z_bc is not None:
            rhs_z = rhs_z + dt * rhs_z_bc

        self._dynamic_noise_runtimes = []
        stochastic_x, runtime_x = build_scalar_state_stochastic_term(
            self,
            state_name="x",
            previous_state=self.x_n,
            test=x_test,
            dt=dt,
        )
        if stochastic_x is not None and runtime_x is not None:
            rhs_x = rhs_x + stochastic_x
            self._dynamic_noise_runtimes.append(runtime_x)
        stochastic_y, runtime_y = build_scalar_state_stochastic_term(
            self,
            state_name="y",
            previous_state=self.y_n,
            test=y_test,
            dt=dt,
        )
        if stochastic_y is not None and runtime_y is not None:
            rhs_y = rhs_y + stochastic_y
            self._dynamic_noise_runtimes.append(runtime_y)
        stochastic_z, runtime_z = build_scalar_state_stochastic_term(
            self,
            state_name="z",
            previous_state=self.z_n,
            test=z_test,
            dt=dt,
        )
        if stochastic_z is not None and runtime_z is not None:
            rhs_z = rhs_z + stochastic_z
            self._dynamic_noise_runtimes.append(runtime_z)

        self._x_problem = self.create_linear_problem(
            lhs_x,
            rhs_x,
            u=self.x_h,
            bcs=x_bcs,
            petsc_options_prefix="plm_lorenz_x_",
            mpc=x_mpc,
        )
        self._y_problem = self.create_linear_problem(
            lhs_y,
            rhs_y,
            u=self.y_h,
            bcs=y_bcs,
            petsc_options_prefix="plm_lorenz_y_",
            mpc=y_mpc,
        )
        self._z_problem = self.create_linear_problem(
            lhs_z,
            rhs_z,
            u=self.z_h,
            bcs=z_bcs,
            petsc_options_prefix="plm_lorenz_z_",
            mpc=z_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._x_problem.solve()
        self._y_problem.solve()
        self._z_problem.solve()

        self.x_n.x.array[:] = self.x_h.x.array
        self.x_n.x.scatter_forward()
        self.y_n.x.array[:] = self.y_h.x.array
        self.y_n.x.scatter_forward()
        self.z_n.x.array[:] = self.z_h.x.array
        self.z_n.x.scatter_forward()

        return (
            self._x_problem.solver.getConvergedReason() > 0
            and self._y_problem.solver.getConvergedReason() > 0
            and self._z_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"x": self.x_n, "y": self.y_n, "z": self.z_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


class LorenzPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _LorenzProblem(self.spec, config)
