"""Transient Maxwell-inspired electric-field pulse propagation."""

import math

import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.initial_conditions import apply_vector_ic
from plm_data.core.mesh import create_domain
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.metadata import (
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
)
from plm_data.presets.physics._maxwell_common import (
    apply_maxwell_dirichlet_bcs,
    build_absorbing_boundary_form,
)

_MAXWELL_PULSE_SPEC = PresetSpec(
    name="maxwell_pulse",
    category="physics",
    description=(
        "Transient electric-field pulse propagation using a curl-curl wave "
        "equation with absorbing and PEC boundaries."
    ),
    equations={
        "electric_field": (
            "epsilon_r * d2E/dt2 + sigma * dE/dt + curl(mu_r^-1 curl(E)) = J(t, x)"
        ),
    },
    parameters=[
        PDEParameter("epsilon_r", "Relative permittivity"),
        PDEParameter("mu_r", "Relative permeability"),
        PDEParameter("sigma", "Ohmic damping coefficient"),
        PDEParameter("pulse_amplitude", "Pulse amplitude"),
        PDEParameter("pulse_frequency", "Pulse carrier frequency"),
        PDEParameter("pulse_width", "Gaussian pulse width"),
        PDEParameter("pulse_delay", "Pulse center time"),
    ],
    inputs={
        "electric_field": InputSpec(
            name="electric_field",
            shape="vector",
            allow_boundary_conditions=True,
            allow_source=True,
            allow_initial_condition=True,
        ),
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
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _MaxwellPulseProblem(TransientLinearProblem):
    def setup(self) -> None:
        domain_geom = create_domain(self.config.domain)
        self.domain_geom = domain_geom
        self.msh = domain_geom.mesh

        field_config = self.config.input("electric_field")
        V = fem.functionspace(self.msh, ("N1curl", 1))
        self._V = V

        self.E_prev = fem.Function(V, name="electric_field_prev")
        self.E_curr = fem.Function(V, name="electric_field")
        self.E_next = fem.Function(V, name="electric_field_next")

        assert field_config.initial_condition is not None
        apply_vector_ic(
            self.E_curr,
            field_config.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.E_curr.x.scatter_forward()
        self.E_prev.x.array[:] = self.E_curr.x.array
        self.E_prev.x.scatter_forward()

        dt = self.config.time.dt
        epsilon_r = self.config.parameters["epsilon_r"]
        mu_r = self.config.parameters["mu_r"]
        sigma = self.config.parameters["sigma"]
        beta = math.sqrt(epsilon_r / mu_r)

        E = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        inv_mu = 1.0 / mu_r

        a = (epsilon_r / dt**2 + sigma / dt) * ufl.inner(
            E, v
        ) * ufl.dx + inv_mu * ufl.inner(ufl.curl(E), ufl.curl(v)) * ufl.dx
        L = (2.0 * epsilon_r / dt**2 + sigma / dt) * ufl.inner(
            self.E_curr, v
        ) * ufl.dx - (epsilon_r / dt**2) * ufl.inner(self.E_prev, v) * ufl.dx

        a_absorb = build_absorbing_boundary_form(
            E,
            v,
            domain_geom,
            field_config.boundary_conditions,
            beta / dt,
            self.config.parameters,
        )
        if a_absorb is not None:
            a = a + a_absorb
            L = L + build_absorbing_boundary_form(
                self.E_curr,
                v,
                domain_geom,
                field_config.boundary_conditions,
                beta / dt,
                self.config.parameters,
            )

        self._pulse_scale = fem.Constant(self.msh, default_scalar_type(0.0))
        assert field_config.source is not None
        source_form = build_vector_source_form(
            v,
            self.msh,
            field_config.source,
            self.config.parameters,
        )
        if source_form is not None:
            L = L + self._pulse_scale * source_form

        bcs = apply_maxwell_dirichlet_bcs(
            V,
            domain_geom,
            field_config.boundary_conditions,
            self.config.parameters,
        )
        self.problem = LinearProblem(
            a,
            L,
            u=self.E_next,
            bcs=bcs,
            petsc_options_prefix="plm_maxwell_pulse_",
            petsc_options=self._solver_options,
        )

    def _pulse_value(self, t: float) -> float:
        p = self.config.parameters
        width = p["pulse_width"]
        envelope = math.exp(-(((t - p["pulse_delay"]) / width) ** 2))
        carrier = math.sin(2.0 * math.pi * p["pulse_frequency"] * t)
        return p["pulse_amplitude"] * envelope * carrier

    def step(self, t: float, dt: float) -> bool:
        self._pulse_scale.value = default_scalar_type(self._pulse_value(t + dt))
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason() > 0
        if converged:
            self.E_prev.x.array[:] = self.E_curr.x.array
            self.E_curr.x.array[:] = self.E_next.x.array
            self.E_prev.x.scatter_forward()
            self.E_curr.x.scatter_forward()
        return converged

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"electric_field": self.E_curr}

    def get_num_dofs(self) -> int:
        return self._V.dofmap.index_map.size_global * self._V.dofmap.index_map_bs


@register_preset("maxwell_pulse")
class MaxwellPulsePreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _MAXWELL_PULSE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _MaxwellPulseProblem(self.spec, config)
