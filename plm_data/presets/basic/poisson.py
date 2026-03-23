"""Poisson equation preset: -div(kappa * grad(u)) = f."""

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import apply_dirichlet_bcs, build_natural_bc_forms
from plm_data.core.config import SimulationConfig
from plm_data.core.mesh import DomainGeometry
from plm_data.core.source_terms import build_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import SteadyLinearPreset
from plm_data.presets.metadata import PDEMetadata, PDEParameter


@register_preset("poisson")
class PoissonPreset(SteadyLinearPreset):

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="poisson",
            category="basic",
            description="Poisson equation -div(kappa * grad(u)) = f",
            equations={"u": "-∇·(κ ∇u) = f"},
            parameters=[
                PDEParameter("kappa", "Diffusion coefficient"),
                PDEParameter("f_amplitude", "Source term amplitude"),
            ],
            field_names=["u"],
            steady_state=True,
            supported_dimensions=[2, 3],
        )

    def create_function_space(self, domain_geom, config):
        return fem.functionspace(domain_geom.mesh, ("Lagrange", 2))

    def create_boundary_conditions(self, V, domain_geom, config):
        return apply_dirichlet_bcs(
            V, domain_geom, config.domain.boundary_conditions, config.parameters
        )

    def create_forms(self, V, domain_geom, config):
        kappa = config.parameters["kappa"]

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        L = build_source_form(v, domain_geom.mesh, config.source_term, config.parameters)
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, config.domain.boundary_conditions, config.parameters
        )
        if a_bc is not None:
            a = a + a_bc
        if L_bc is not None:
            L = L + L_bc

        return a, L
