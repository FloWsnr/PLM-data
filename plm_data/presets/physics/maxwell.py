"""Time-harmonic Maxwell equations in curl-conforming spaces."""

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem

from plm_data.core.periodic import require_unverified_periodic_support
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
    apply_split_maxwell_dirichlet_bcs,
    build_absorbing_boundary_form,
)

_MAXWELL_SPEC = PresetSpec(
    name="maxwell",
    category="physics",
    description=(
        "Time-harmonic Maxwell curl-curl solve for the electric field in "
        "homogeneous media with PEC and absorbing boundaries, using a "
        "real/imaginary split in a mixed H(curl) space."
    ),
    equations={
        "electric_field_real": (
            "curl(mu_r^-1 curl(E_r)) - k0^2 epsilon_r E_r - kappa T(E_i) = J"
        ),
        "electric_field_imag": (
            "curl(mu_r^-1 curl(E_i)) - k0^2 epsilon_r E_i + kappa T(E_r) = 0"
        ),
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
    states={
        "electric_field_real": StateSpec(name="electric_field_real", shape="vector"),
        "electric_field_imag": StateSpec(name="electric_field_imag", shape="vector"),
    },
    outputs={
        "electric_field_real": OutputSpec(
            name="electric_field_real",
            shape="vector",
            output_mode="components",
            source_name="electric_field_real",
        ),
        "electric_field_imag": OutputSpec(
            name="electric_field_imag",
            shape="vector",
            output_mode="components",
            source_name="electric_field_imag",
        ),
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
        nedelec = element(
            "N1curl",
            domain_geom.mesh.basix_cell(),
            1,
            dtype=default_real_type,
        )
        return fem.functionspace(domain_geom.mesh, mixed_element([nedelec, nedelec]))

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
        return apply_split_maxwell_dirichlet_bcs(
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

        E_r, E_i = ufl.TrialFunctions(V)
        v_r, v_i = ufl.TestFunctions(V)
        inv_mu = 1.0 / mu_r

        def _volume_form(trial, test):
            return (
                inv_mu * ufl.inner(ufl.curl(trial), ufl.curl(test)) * ufl.dx
                - (k0**2) * epsilon_r * ufl.inner(trial, test) * ufl.dx
            )

        a = _volume_form(E_r, v_r) + _volume_form(E_i, v_i)

        source_form = None
        if field_config.source is not None:
            source_form = build_vector_source_form(
                v_r,
                domain_geom.mesh,
                field_config.source,
                self.config.parameters,
            )
        if source_form is None:
            zero = fem.Constant(
                domain_geom.mesh,
                np.zeros(domain_geom.mesh.geometry.dim, dtype=default_real_type),
            )
            L = ufl.inner(zero, v_r) * ufl.dx
        else:
            L = source_form

        zero = fem.Constant(
            domain_geom.mesh,
            np.zeros(domain_geom.mesh.geometry.dim, dtype=default_real_type),
        )
        L = L + ufl.inner(zero, v_i) * ufl.dx

        absorbing_coeff = k0 * np.sqrt(epsilon_r / mu_r)
        a_rr = build_absorbing_boundary_form(
            E_r,
            v_i,
            domain_geom,
            boundary_field,
            absorbing_coeff,
            self.config.parameters,
        )
        if a_rr is not None:
            a = a + a_rr

        a_ir = build_absorbing_boundary_form(
            E_i,
            v_r,
            domain_geom,
            boundary_field,
            absorbing_coeff,
            self.config.parameters,
        )
        if a_ir is not None:
            a = a - a_ir

        return a, L

    def export_solution_fields(self, solution):
        V_r, real_dofs = solution.function_space.sub(0).collapse()
        E_r = fem.Function(V_r, name="electric_field_real")
        E_r.x.array[:] = solution.x.array[real_dofs]
        E_r.x.scatter_forward()

        V_i, imag_dofs = solution.function_space.sub(1).collapse()
        E_i = fem.Function(V_i, name="electric_field_imag")
        E_i.x.array[:] = solution.x.array[imag_dofs]
        E_i.x.scatter_forward()

        return {
            "electric_field_real": E_r,
            "electric_field_imag": E_i,
        }


@register_preset("maxwell")
class MaxwellPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _MAXWELL_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _MaxwellProblem(self.spec, config)
