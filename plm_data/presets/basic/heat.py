"""Heat equation preset: du/dt = kappa * laplacian(u) + f."""

import numpy as np
import ufl
from dolfinx import fem
from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_domain
from plm_data.core.source_terms import build_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.metadata import (
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
)

_HEAT_SPEC = PresetSpec(
    name="heat",
    category="basic",
    description="Heat equation du/dt = kappa * laplacian(u) + f",
    equations={"u": "∂u/∂t = κ ∇²u + f"},
    parameters=[PDEParameter("kappa", "Thermal diffusivity")],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_boundary_conditions=True,
            allow_source=True,
            allow_initial_condition=True,
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
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _HeatProblem(TransientLinearProblem):
    def setup(self) -> None:
        domain_geom = create_domain(self.config.domain)
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        kappa = self.config.parameters["kappa"]
        field_config = self.config.input("u")
        dt = self.config.time.dt
        bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            field_config.boundary_conditions,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(V, domain_geom, bcs)
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
        kappa_c = fem.Constant(self.msh, np.float64(kappa))

        a = (
            ufl.inner(u, v) * ufl.dx
            + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
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
            field_config.boundary_conditions,
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
