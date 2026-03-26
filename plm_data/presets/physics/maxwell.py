"""Time-harmonic Maxwell equations in curl-conforming spaces."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem

from plm_data.core.runtime import require_complex_runtime
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, StationaryLinearProblem
from plm_data.presets.metadata import FieldSpec, PDEParameter, PresetSpec
from plm_data.presets.physics._maxwell_common import (
    apply_maxwell_dirichlet_bcs,
    build_absorbing_boundary_form,
)

_MAXWELL_SPEC = PresetSpec(
    name="maxwell",
    category="physics",
    description=(
        "Time-harmonic Maxwell curl-curl solve for the electric field in "
        "homogeneous media with PEC and absorbing boundaries."
    ),
    equations={
        "electric_field": "curl(mu_r^-1 curl(E)) - k0^2 epsilon_r E = J",
    },
    parameters=[
        PDEParameter("epsilon_r", "Relative permittivity"),
        PDEParameter("mu_r", "Relative permeability"),
        PDEParameter("k0", "Background wavenumber"),
        PDEParameter("source_amplitude", "Interior source amplitude"),
    ],
    fields={
        "electric_field": FieldSpec(
            name="electric_field",
            shape="vector",
            allow_boundary_conditions=True,
            allow_source=True,
            allow_initial_condition=False,
            output_mode="components",
        ),
    },
    family="stationary_linear",
    steady_state=True,
    supported_dimensions=[2, 3],
)


class _MaxwellProblem(StationaryLinearProblem):
    def create_function_space(self, domain_geom):
        return fem.functionspace(domain_geom.mesh, ("N1curl", 1))

    def create_boundary_conditions(self, V, domain_geom):
        return apply_maxwell_dirichlet_bcs(
            V,
            domain_geom,
            self.config.field("electric_field").boundary_conditions,
            self.config.parameters,
        )

    def create_forms(self, V, domain_geom):
        epsilon_r = self.config.parameters["epsilon_r"]
        mu_r = self.config.parameters["mu_r"]
        k0 = self.config.parameters["k0"]
        field_config = self.config.field("electric_field")

        E = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        inv_mu = 1.0 / mu_r
        a = (
            inv_mu * ufl.inner(ufl.curl(E), ufl.curl(v)) * ufl.dx
            - (k0**2) * epsilon_r * ufl.inner(E, v) * ufl.dx
        )

        source_form = None
        if field_config.source is not None:
            source_form = build_vector_source_form(
                v,
                domain_geom.mesh,
                field_config.source,
                self.config.parameters,
            )
        if source_form is None:
            zero = fem.Constant(
                domain_geom.mesh,
                np.zeros(domain_geom.mesh.geometry.dim, dtype=default_scalar_type),
            )
            L = ufl.inner(zero, v) * ufl.dx
        else:
            L = source_form

        absorbing_coeff = 1j * k0 * np.sqrt(epsilon_r / mu_r)
        a_absorb = build_absorbing_boundary_form(
            E,
            v,
            domain_geom,
            field_config.boundary_conditions,
            absorbing_coeff,
            self.config.parameters,
        )
        if a_absorb is not None:
            a = a + a_absorb

        return a, L

    def export_solution_fields(self, solution):
        solution.name = "electric_field"
        return {"electric_field": solution}


@register_preset("maxwell")
class MaxwellPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _MAXWELL_SPEC

    def build_problem(self, config) -> ProblemInstance:
        require_complex_runtime(self.spec.name)
        return _MaxwellProblem(self.spec, config)
