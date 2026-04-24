"""Heat equation PDE: du/dt = div(kappa * grad(u)) + f."""

import numpy as np
import ufl
from dolfinx import fem
from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.core.stochastic import (
    build_scalar_coefficient,
    build_scalar_state_stochastic_term,
)
from plm_data.core.source_terms import build_source_form
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_HEAT_SPEC = PDESpec(
    name="heat",
    category="basic",
    description="Heat equation du/dt = div(kappa * grad(u)) + f",
    equations={"u": "∂u/∂t = ∇·(κ ∇u) + f"},
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
            description="Boundary conditions for the scalar temperature field.",
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
    supported_dimensions=[2],
    coefficients={
        "kappa": CoefficientSpec(
            name="kappa",
            shape="scalar",
            description="Thermal diffusivity coefficient field.",
            allow_randomization=True,
        )
    },
)


class _HeatProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
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

        self.u_n = fem.Function(self.V, name="u_n")
        self.uh = fem.Function(self.V, name="u")

        assert field_config.initial_condition is not None
        apply_ic(
            self.u_n,
            field_config.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        dt_c = fem.Constant(self.msh, np.float64(dt))
        kappa = build_scalar_coefficient(self, "kappa")
        if kappa is None:
            raise ValueError("Heat coefficient 'kappa' cannot use a custom expression")

        a = (
            ufl.inner(u, v) * ufl.dx
            + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        L = ufl.inner(self.u_n, v) * ufl.dx

        assert field_config.source is not None
        source = build_source_form(
            v,
            self.msh,
            field_config.source,
            self.config.parameters,
        )
        if source is not None:
            L = L + dt_c * source

        stochastic_term, stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=v,
            dt=dt,
        )
        self._dynamic_noise_runtimes = []
        if stochastic_term is not None and stochastic_runtime is not None:
            L = L + stochastic_term
            self._dynamic_noise_runtimes.append(stochastic_runtime)

        a_bc, L_bc = build_natural_bc_forms(
            u,
            v,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        if a_bc is not None:
            a = a + dt_c * a_bc
        if L_bc is not None:
            L = L + dt_c * L_bc

        self.problem = self.create_linear_problem(
            a,
            L,
            u=self.uh,
            bcs=bcs,
            petsc_options_prefix="plm_heat_",
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
        return self.V.dofmap.index_map.size_global


class HeatPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _HEAT_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _HeatProblem(self.spec, config)
