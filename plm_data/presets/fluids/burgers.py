"""Vector viscous Burgers equation preset."""

import ufl
from dolfinx import default_real_type, fem

from plm_data.core.boundary_conditions import (
    apply_vector_dirichlet_bcs,
    build_vector_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_vector_ic
from plm_data.core.solver_strategies import NONLINEAR_MIXED_DIRECT
from plm_data.core.stochastic import build_vector_state_stochastic_term
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientNonlinearProblem
from plm_data.presets.boundary_validation import (
    validate_vector_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_BURGERS_SPEC = PresetSpec(
    name="burgers",
    category="fluids",
    description=(
        "Vector viscous Burgers equation with nonlinear self-advection and diffusion."
    ),
    equations={
        "velocity": "du/dt + (u·grad)u = nu * laplacian(u) + f",
    },
    parameters=[
        PDEParameter("nu", "Kinematic viscosity / diffusion coefficient"),
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
            description="Boundary conditions for the Burgers velocity field.",
        ),
    },
    states={
        "velocity": StateSpec(
            name="velocity",
            shape="vector",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _BurgersProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="velocity",
            boundary_field=self.config.boundary_field("velocity"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        gdim = self.msh.geometry.dim

        dt = self.config.time.dt
        nu = self.config.parameters["nu"]
        velocity_input = self.config.input("velocity")
        velocity_boundary_field = self.config.boundary_field("velocity")

        V = fem.functionspace(self.msh, ("Lagrange", 1, (gdim,)))
        bcs = apply_vector_dirichlet_bcs(
            V,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(
            V, domain_geom, velocity_boundary_field, bcs
        )
        solution_space = V if mpc is None else mpc.function_space

        self.u = fem.Function(solution_space, name="velocity")
        self.u_n = fem.Function(solution_space, name="velocity_prev")
        self._num_dofs = (
            solution_space.dofmap.index_map.size_global
            * solution_space.dofmap.index_map_bs
        )

        assert velocity_input.initial_condition is not None
        apply_vector_ic(
            self.u,
            velocity_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.u.x.scatter_forward()
        self.u_n.x.array[:] = self.u.x.array
        self.u_n.x.scatter_forward()

        v = ufl.TestFunction(V)
        dt_c = fem.Constant(self.msh, default_real_type(dt))
        nu_c = fem.Constant(self.msh, default_real_type(nu))

        F = (
            ufl.inner((self.u - self.u_n) / dt_c, v) * ufl.dx
            + ufl.inner(ufl.dot(ufl.grad(self.u), self.u), v) * ufl.dx
            + nu_c * ufl.inner(ufl.grad(self.u), ufl.grad(v)) * ufl.dx
        )

        assert velocity_input.source is not None
        source_form = build_vector_source_form(
            v,
            self.msh,
            velocity_input.source,
            self.config.parameters,
        )
        if source_form is not None:
            F = F - source_form

        stochastic_term, stochastic_runtime = build_vector_state_stochastic_term(
            self,
            state_name="velocity",
            previous_state=self.u_n,
            test=v,
            dt=dt,
        )
        self._dynamic_noise_runtimes = []
        if stochastic_term is not None and stochastic_runtime is not None:
            F = F - stochastic_term
            self._dynamic_noise_runtimes.append(stochastic_runtime)

        traction_form = build_vector_natural_bc_forms(
            v,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
        )
        if traction_form is not None:
            F = F - traction_form

        J = ufl.derivative(F, self.u, ufl.TrialFunction(V))
        self.problem = self.create_nonlinear_problem(
            F,
            self.u,
            bcs=bcs,
            petsc_options_prefix="plm_burgers_",
            J=J,
            mpc=mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self.problem.solve()
        self.u.x.scatter_forward()
        self.u_n.x.array[:] = self.u.x.array
        self.u_n.x.scatter_forward()
        return self.problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"velocity": self.u}

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("burgers")
class BurgersPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _BURGERS_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _BurgersProblem(self.spec, config)
