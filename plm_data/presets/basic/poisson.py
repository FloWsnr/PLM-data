"""Poisson equation preset: -div(kappa * grad(u)) = f."""

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.source_terms import build_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, StationaryLinearProblem
from plm_data.presets.metadata import (
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
)

_POISSON_SPEC = PresetSpec(
    name="poisson",
    category="basic",
    description="Poisson equation -div(kappa * grad(u)) = f",
    equations={"u": "-∇·(κ ∇u) = f"},
    parameters=[
        PDEParameter("kappa", "Diffusion coefficient"),
        PDEParameter("f_amplitude", "Source term amplitude"),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_boundary_conditions=True,
            allow_source=True,
            allow_initial_condition=False,
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
    steady_state=True,
    supported_dimensions=[2, 3],
)


class _PoissonProblem(StationaryLinearProblem):
    def create_function_space(self, domain_geom):
        return fem.functionspace(domain_geom.mesh, ("Lagrange", 2))

    def create_boundary_conditions(self, V, domain_geom):
        return apply_dirichlet_bcs(
            V,
            domain_geom,
            self.config.input("u").boundary_conditions,
            self.config.parameters,
        )

    def create_forms(self, V, domain_geom):
        kappa = self.config.parameters["kappa"]
        field_config = self.config.input("u")

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        zero = fem.Constant(domain_geom.mesh, 0.0)

        assert field_config.source is not None
        L = build_source_form(
            v,
            domain_geom.mesh,
            field_config.source,
            self.config.parameters,
        )
        if L is None:
            L = ufl.inner(zero, v) * ufl.dx
        a_bc, L_bc = build_natural_bc_forms(
            u,
            v,
            domain_geom,
            field_config.boundary_conditions,
            self.config.parameters,
        )
        if a_bc is not None:
            a = a + a_bc
        if L_bc is not None:
            L = L + L_bc

        return a, L

    def export_solution_fields(self, solution):
        solution.name = "u"
        return {"u": solution}


@register_preset("poisson")
class PoissonPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _POISSON_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _PoissonProblem(self.spec, config)
