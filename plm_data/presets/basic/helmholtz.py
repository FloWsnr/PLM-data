"""Helmholtz equation preset: -div(kappa * grad(u)) - k^2 * u = f."""

import logging

import numpy as np
import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.solver_strategies import STATIONARY_INDEFINITE_DIRECT
from plm_data.core.source_terms import build_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, StationaryLinearProblem
from plm_data.presets.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

logger = logging.getLogger(__name__)

_HELMHOLTZ_SPEC = PresetSpec(
    name="helmholtz",
    category="basic",
    description="Helmholtz equation -div(kappa * grad(u)) - k^2 * u = f",
    equations={"u": "-∇·(κ ∇u) - k²u = f"},
    parameters=[
        PDEParameter("kappa", "Diffusion coefficient"),
        PDEParameter("k", "Wavenumber"),
        PDEParameter("f_amplitude", "Source term amplitude"),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=False,
        )
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the scalar solution.",
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
    steady_state=True,
    supported_dimensions=[2, 3],
)


def _check_resonance(k: float, domain_size: list[float]) -> None:
    """Warn if k^2 is close to an eigenvalue of -nabla^2 on the domain.

    On a rectangle/box [0, L1] x [0, L2] x ... with Dirichlet BCs, the
    eigenvalues are pi^2 * sum(m_i^2 / L_i^2) for positive integers m_i.
    When k^2 approaches an eigenvalue, the problem becomes ill-conditioned.
    """
    k_sq = k**2
    # Check low-order eigenvalues (m_i = 1..4 in each dimension)
    modes = range(1, 5)
    for combo in np.ndindex(*(len(list(modes)),) * len(domain_size)):
        m = [list(modes)[i] for i in combo]
        eigenvalue = np.pi**2 * sum((mi / Li) ** 2 for mi, Li in zip(m, domain_size))
        if abs(k_sq - eigenvalue) < 0.5:
            mode_str = ",".join(str(mi) for mi in m)
            logger.warning(
                "Wavenumber k=%.4g (k^2=%.4g) is near eigenvalue %.4g "
                "(mode %s). The problem may be ill-conditioned.",
                k,
                k_sq,
                eigenvalue,
                mode_str,
            )
            return


class _HelmholtzProblem(StationaryLinearProblem):
    supported_solver_strategies = (STATIONARY_INDEFINITE_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )

    def create_function_space(self, domain_geom):
        return fem.functionspace(domain_geom.mesh, ("Lagrange", 2))

    def periodic_boundary_field(self):
        return self.config.boundary_field("u")

    def create_boundary_conditions(self, V, domain_geom):
        return apply_dirichlet_bcs(
            V,
            domain_geom,
            self.config.boundary_field("u"),
            self.config.parameters,
        )

    def create_forms(self, V, domain_geom):
        kappa = self.config.parameters["kappa"]
        k = self.config.parameters["k"]
        field_config = self.config.input("u")

        domain_size = self.config.domain.params.get("size")
        if domain_size is not None:
            _check_resonance(k, domain_size)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = (
            kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - k**2 * ufl.inner(u, v) * ufl.dx
        )
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
            self.config.boundary_field("u"),
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


@register_preset("helmholtz")
class HelmholtzPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _HELMHOLTZ_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _HelmholtzProblem(self.spec, config)
