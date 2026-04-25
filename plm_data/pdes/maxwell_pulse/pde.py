"""Transient Maxwell-inspired electric-field pulse propagation."""

import math

import ufl
from dolfinx import default_scalar_type, fem

from plm_data.boundary_conditions.runtime import apply_vector_dirichlet_bcs
from plm_data.initial_conditions.runtime import apply_vector_ic
from plm_data.boundary_conditions.periodic import require_unverified_periodic_support
from plm_data.core.runtime_config import BoundaryFieldConfig, FieldExpressionConfig
from plm_data.core.solver_strategies import CONSTANT_LHS_CURL_DIRECT
from plm_data.core.source_terms import build_vector_source_form
from plm_data.core.spatial_fields import (
    component_expressions,
    component_labels_for_dim,
    resolve_param_ref,
    scalar_expression_to_config,
)
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import validate_vector_standard_boundary_field
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    MAXWELL_BOUNDARY_OPERATORS,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_MAXWELL_PULSE_SPEC = PDESpec(
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
            allow_source=True,
            allow_initial_condition=True,
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
    supported_dimensions=[2],
)


def _as_3d_vector(value, gdim: int):
    if gdim == 2:
        return ufl.as_vector((value[0], value[1], 0))
    return value


def _tangential_inner(u, v, n, gdim: int):
    u_3d = _as_3d_vector(u, gdim)
    v_3d = _as_3d_vector(v, gdim)
    n_3d = _as_3d_vector(n, gdim)
    return ufl.inner(ufl.cross(u_3d, n_3d), ufl.cross(v_3d, n_3d))


def _validate_boundary_types(
    boundary_field: BoundaryFieldConfig,
    allowed: set[str],
) -> None:
    for name, entries in boundary_field.sides.items():
        for bc in entries:
            if bc.type not in allowed:
                allowed_str = ", ".join(sorted(allowed))
                raise ValueError(
                    f"Maxwell boundary '{name}' uses unsupported type '{bc.type}'. "
                    f"Allowed types: {allowed_str}."
                )


def _require_zero_absorbing_value(
    name: str,
    expr: FieldExpressionConfig,
    gdim: int,
    parameters: dict[str, float],
) -> None:
    components = component_expressions(expr, gdim)
    for label in component_labels_for_dim(gdim):
        component = scalar_expression_to_config(components[label])
        expr_type = component["type"]
        if expr_type in {"none", "zero"}:
            continue
        if expr_type == "constant":
            value = resolve_param_ref(component["params"]["value"], parameters)
            if value == 0.0:
                continue
        raise ValueError(
            f"Absorbing boundary '{name}' must use a zero value. "
            f"Component '{label}' was configured with '{expr_type}'."
        )


def _build_absorbing_boundary_form(
    u,
    v,
    domain_geom,
    boundary_field: BoundaryFieldConfig,
    coefficient,
    parameters: dict[str, float],
):
    _validate_boundary_types(
        boundary_field,
        allowed={"dirichlet", "absorbing", "periodic"},
    )

    gdim = domain_geom.mesh.geometry.dim
    n = ufl.FacetNormal(domain_geom.mesh)
    form = None
    for name, entries in boundary_field.sides.items():
        for bc in entries:
            if bc.type != "absorbing":
                continue
            assert bc.value is not None
            _require_zero_absorbing_value(name, bc.value, gdim, parameters)
            term = (
                coefficient
                * _tangential_inner(u, v, n, gdim)
                * domain_geom.ds(domain_geom.boundary_names[name])
            )
            form = term if form is None else form + term
    return form


class _MaxwellPulseProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_CURL_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        validate_vector_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="electric_field",
            boundary_field=self.config.boundary_field("electric_field"),
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "absorbing", "periodic"},
        )

    def setup(self) -> None:
        assert self.config.time is not None
        domain_geom = self.load_domain_geometry()
        boundary_field = self.config.boundary_field("electric_field")
        require_unverified_periodic_support(
            self.spec.name,
            boundary_field,
            "N1curl spaces",
        )
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
        if min(epsilon_r, mu_r) <= 0.0 or sigma < 0.0:
            raise ValueError(
                "Maxwell pulse requires epsilon_r and mu_r > 0, and sigma >= 0."
            )
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

        a_absorb = _build_absorbing_boundary_form(
            E,
            v,
            domain_geom,
            boundary_field,
            beta / dt,
            self.config.parameters,
        )
        if a_absorb is not None:
            a = a + a_absorb
            L = L + _build_absorbing_boundary_form(
                self.E_curr,
                v,
                domain_geom,
                boundary_field,
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

        _validate_boundary_types(
            boundary_field,
            allowed={"dirichlet", "absorbing", "periodic"},
        )
        bcs = apply_vector_dirichlet_bcs(
            V,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        self.problem = self.create_linear_problem(
            a,
            L,
            u=self.E_next,
            bcs=bcs,
            petsc_options_prefix="plm_maxwell_pulse_",
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


class MaxwellPulsePDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _MAXWELL_PULSE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _MaxwellPulseProblem(self.spec, config)
