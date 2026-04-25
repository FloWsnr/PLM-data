"""Kuramoto-Sivashinsky equation PDE."""

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
    PDESpec,
)
from plm_data.pdes.kuramoto_sivashinsky.spec import PDE_SPEC


def _velocity_vector(msh, parameters: dict[str, float]):
    gdim = msh.geometry.dim
    components = [parameters["advection_x"], parameters["advection_y"]]
    if gdim == 3:
        components.append(parameters["advection_z"])
    return ufl.as_vector(components[:gdim])


def _has_nonzero_advection(msh, parameters: dict[str, float]) -> bool:
    component_names = ("advection_x", "advection_y", "advection_z")[: msh.geometry.dim]
    return any(abs(parameters[name]) > 0.0 for name in component_names)


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
                    "Kuramoto-Sivashinsky wall boundary operators require an "
                    f"explicit zero value. Side '{side_name}' is missing one."
                )
            if not is_exact_zero_field_expression(entry.value, parameters):
                raise ValueError(
                    "Kuramoto-Sivashinsky currently supports only homogeneous "
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


class _KuramotoSivashinskyProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        boundary_field = self.config.boundary_field("u")
        validate_boundary_field_structure(
            pde_name=self.spec.name,
            field_name="u",
            boundary_field=boundary_field,
            domain_geom=domain_geom,
            allowed_operators=set(self.spec.boundary_fields["u"].operators),
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
        params = self.config.parameters
        theta = params["theta"]
        hyperdiffusion = params["hyperdiffusion"]
        anti_diffusion = params["anti_diffusion"]
        nonlinear_strength = params["nonlinear_strength"]
        damping = params["damping"]
        dt = self.config.time.dt
        if min(hyperdiffusion, anti_diffusion, nonlinear_strength) <= 0.0:
            raise ValueError(
                "Kuramoto-Sivashinsky hyperdiffusion, anti_diffusion, and "
                "nonlinear_strength must be positive."
            )
        if damping < 0.0:
            raise ValueError("Kuramoto-Sivashinsky damping cannot be negative.")
        if not 0.0 <= theta <= 1.0:
            raise ValueError("Kuramoto-Sivashinsky theta must be between 0 and 1.")
        velocity = _velocity_vector(self.msh, params)

        bcs = _build_zero_dirichlet_bcs(ME, domain_geom, boundary_field, params)
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

        u, v = ufl.split(self.u)
        u0, v0 = ufl.split(self.u0)
        q, w = ufl.TestFunctions(ME)

        u_theta = (1.0 - theta) * u0 + theta * u
        v_theta = (1.0 - theta) * v0 + theta * v
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        F0 = (
            ufl.inner(u, q) * ufl.dx
            - ufl.inner(u0, q) * ufl.dx
            + dt_c * hyperdiffusion * ufl.inner(ufl.grad(v_theta), ufl.grad(q)) * ufl.dx
            - dt_c * anti_diffusion * ufl.inner(v_theta, q) * ufl.dx
            + dt_c * damping * ufl.inner(u_theta, q) * ufl.dx
            + dt_c
            * 0.5
            * nonlinear_strength
            * ufl.inner(ufl.grad(u), ufl.grad(u))
            * q
            * ufl.dx
        )
        if _has_nonzero_advection(self.msh, params):
            F0 += dt_c * ufl.inner(velocity, ufl.grad(u_theta)) * q * ufl.dx
        F1 = ufl.inner(v, w) * ufl.dx - ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
        F = F0 + F1
        J = ufl.derivative(F, self.u, ufl.TrialFunction(ME))

        self.problem = self.create_nonlinear_problem(
            F,
            self.u,
            bcs=bcs,
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


class KuramotoSivashinskyPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _KuramotoSivashinskyProblem(self.spec, config)
