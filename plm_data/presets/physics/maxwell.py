"""Time-harmonic Maxwell equations in curl-conforming spaces."""

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem

from plm_data.core.periodic import require_unverified_periodic_support
from plm_data.core.runtime import require_complex_runtime
from plm_data.core.solver_strategies import STATIONARY_INDEFINITE_DIRECT
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, StationaryLinearProblem
from plm_data.presets.boundary_validation import (
    validate_vector_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    MAXWELL_BOUNDARY_OPERATORS,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
)
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
    inputs={
        "electric_field": InputSpec(
            name="electric_field",
            shape="vector",
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "electric_field": BoundaryFieldSpec(
            name="electric_field",
            shape="vector",
            operators=MAXWELL_BOUNDARY_OPERATORS,
            description="Boundary conditions for the electric field.",
        )
    },
    states={"electric_field": StateSpec(name="electric_field", shape="vector")},
    outputs={
        "electric_field": OutputSpec(
            name="electric_field",
            shape="vector",
            output_mode="components",
            source_name="electric_field",
        )
    },
    static_fields=[],
    steady_state=True,
    supported_dimensions=[2, 3],
)


class _MaxwellProblem(StationaryLinearProblem):
    supported_solver_strategies = (STATIONARY_INDEFINITE_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="electric_field",
            boundary_field=self.config.boundary_field("electric_field"),
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "absorbing", "periodic"},
        )

    def create_function_space(self, domain_geom):
        return fem.functionspace(domain_geom.mesh, ("N1curl", 1))

    def periodic_boundary_field(self):
        return self.config.boundary_field("electric_field")

    def create_periodic_constraint(self, V, domain_geom, boundary_field, bcs):
        require_unverified_periodic_support(
            self.spec.name,
            boundary_field,
            "N1curl spaces",
        )
        return None

    def create_boundary_conditions(self, V, domain_geom):
        return apply_maxwell_dirichlet_bcs(
            V,
            domain_geom,
            self.config.boundary_field("electric_field"),
            self.config.parameters,
        )

    def create_forms(self, V, domain_geom):
        epsilon_r = self.config.parameters["epsilon_r"]
        mu_r = self.config.parameters["mu_r"]
        k0 = self.config.parameters["k0"]
        field_config = self.config.input("electric_field")
        boundary_field = self.config.boundary_field("electric_field")

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
            boundary_field,
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
