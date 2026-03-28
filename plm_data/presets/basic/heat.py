"""Heat equation preset: du/dt = div(kappa * grad(u)) + f."""

import numpy as np
import ufl
from dolfinx import fem
from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.spatial_fields import build_ufl_field, scalar_expression_to_config
from plm_data.core.source_terms import build_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_HEAT_SPEC = PresetSpec(
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
    states={"u": StateSpec(name="u", shape="scalar")},
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
        "kappa": CoefficientSpec(
            name="kappa",
            shape="scalar",
            description="Thermal diffusivity coefficient field.",
        )
    },
)


class _HeatProblem(TransientLinearProblem):
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
        kappa_config = self.config.coefficient("kappa")
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
        kappa = build_ufl_field(
            self.msh,
            scalar_expression_to_config(kappa_config),
            self.config.parameters,
        )
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


@register_preset("heat")
class HeatPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _HEAT_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _HeatProblem(self.spec, config)
