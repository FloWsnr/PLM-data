"""Advection-diffusion equation preset with SUPG stabilization."""

import ufl
from dolfinx import default_real_type, fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_NONSYMMETRIC
from plm_data.core.stochastic import (
    build_scalar_coefficient,
    build_scalar_state_stochastic_term,
)
from plm_data.core.spatial_fields import (
    build_ufl_field,
    build_vector_ufl_field,
    scalar_expression_to_config,
)
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_ADVECTION_SPEC = PresetSpec(
    name="advection",
    category="basic",
    description=(
        "Scalar advection-diffusion equation with a prescribed velocity field "
        "and SUPG stabilization."
    ),
    equations={
        "u": "∂u/∂t + v·∇u = ∇·(D ∇u) + f",
    },
    parameters=[],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        )
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the transported scalar field.",
        )
    },
    states={
        "u": StateSpec(
            name="u",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        )
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        )
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
    coefficients={
        "velocity": CoefficientSpec(
            name="velocity",
            shape="vector",
            description="Prescribed advection velocity field.",
        ),
        "diffusivity": CoefficientSpec(
            name="diffusivity",
            shape="scalar",
            description="Scalar diffusion coefficient field.",
            allow_randomization=True,
        ),
    },
)


class _AdvectionProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_NONSYMMETRIC,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        field_config = self.config.input("u")
        boundary_field = self.config.boundary_field("u")
        dt = self.config.time.dt

        bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(V, domain_geom, boundary_field, bcs)
        self.V = V if mpc is None else mpc.function_space
        self._num_dofs = (
            self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        )

        self.u_n = fem.Function(self.V, name="u_n")
        self.uh = fem.Function(self.V, name="u")

        assert field_config.initial_condition is not None
        apply_ic(
            self.u_n,
            field_config.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.u_n.x.scatter_forward()

        velocity = build_vector_ufl_field(
            self.msh,
            self.config.coefficient("velocity"),
            self.config.parameters,
        )
        if velocity is None:
            raise ValueError(
                "Advection coefficient 'velocity' cannot use a custom expression"
            )

        diffusivity = build_scalar_coefficient(self, "diffusivity")
        if diffusivity is None:
            raise ValueError(
                "Advection coefficient 'diffusivity' cannot use a custom expression"
            )

        assert field_config.source is not None
        source = build_ufl_field(
            self.msh,
            scalar_expression_to_config(field_config.source),
            self.config.parameters,
        )
        if source is None:
            raise ValueError("Advection source 'u' cannot use a custom expression")

        u = ufl.TrialFunction(V)
        w = ufl.TestFunction(V)
        dt_c = fem.Constant(self.msh, default_real_type(dt))
        zero = fem.Constant(self.msh, default_real_type(0.0))

        advection_term = ufl.inner(velocity, ufl.grad(u))
        diffusion_term = ufl.inner(diffusivity * ufl.grad(u), ufl.grad(w))
        a = (
            ufl.inner(u, w) * ufl.dx
            + dt_c * (advection_term * w + diffusion_term) * ufl.dx
        )
        L = (ufl.inner(self.u_n, w) + dt_c * ufl.inner(source, w)) * ufl.dx

        stochastic_term, stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=w,
            dt=dt,
        )
        self._dynamic_noise_runtimes = []
        if stochastic_term is not None and stochastic_runtime is not None:
            L = L + stochastic_term
            self._dynamic_noise_runtimes.append(stochastic_runtime)

        a_bc, L_bc = build_natural_bc_forms(
            u,
            w,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        if a_bc is not None:
            a = a + dt_c * a_bc
        if L_bc is not None:
            L = L + dt_c * L_bc

        h = ufl.CellDiameter(self.msh)
        velocity_norm = ufl.sqrt(ufl.inner(velocity, velocity))
        stabilized_diffusivity = ufl.max_value(diffusivity, zero)
        tau = 1.0 / ufl.sqrt(
            (2.0 / dt_c) ** 2
            + (2.0 * velocity_norm / h) ** 2
            + (4.0 * stabilized_diffusivity / (h**2)) ** 2
        )
        streamline_test = ufl.inner(velocity, ufl.grad(w))
        lhs_residual = (
            u / dt_c
            + ufl.inner(velocity, ufl.grad(u))
            - ufl.div(diffusivity * ufl.grad(u))
        )
        rhs_residual = self.u_n / dt_c + source
        a = a + dt_c * tau * lhs_residual * streamline_test * ufl.dx
        L = L + dt_c * tau * rhs_residual * streamline_test * ufl.dx

        self.problem = self.create_linear_problem(
            a,
            L,
            u=self.uh,
            bcs=bcs,
            petsc_options_prefix="plm_advection_",
            mpc=mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self.problem.solve()
        self.u_n.x.array[:] = self.uh.x.array
        self.u_n.x.scatter_forward()
        return self.problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("advection")
class AdvectionPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _ADVECTION_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _AdvectionProblem(self.spec, config)
