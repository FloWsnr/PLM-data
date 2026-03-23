"""Heat equation preset: du/dt = kappa * laplacian(u) + f."""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.config import SimulationConfig
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_domain
from plm_data.core.source_terms import build_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import TimeDependentPreset
from plm_data.presets.metadata import PDEMetadata, PDEParameter


@register_preset("heat")
class HeatPreset(TimeDependentPreset):
    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heat",
            category="basic",
            description="Heat equation du/dt = kappa * laplacian(u) + f",
            equations={"u": "∂u/∂t = κ ∇²u + f"},
            parameters=[
                PDEParameter("kappa", "Thermal diffusivity"),
            ],
            field_names=["u"],
            steady_state=False,
            supported_dimensions=[2, 3],
        )

    def setup(self, config: SimulationConfig) -> None:
        domain_geom = create_domain(config.domain)
        self.msh = domain_geom.mesh
        self.V = fem.functionspace(self.msh, ("Lagrange", 1))

        kappa = config.parameters["kappa"]
        dt = config.dt

        # Current and previous solution
        self.u_n = fem.Function(self.V, name="u_n")
        self.uh = fem.Function(self.V, name="u")

        # Apply initial condition from config
        apply_ic(
            self.u_n,  # type: ignore[reportArgumentType]
            config.initial_conditions["u"],
            config.parameters,
            seed=config.seed,
        )

        # Implicit Euler: (u - u_n)/dt = kappa * laplacian(u) + f
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        dt_c = fem.Constant(self.msh, np.float64(dt))
        kappa_c = fem.Constant(self.msh, np.float64(kappa))

        a = (
            ufl.inner(u, v) * ufl.dx  # type: ignore[reportOperatorIssue]
            + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # type: ignore[reportOperatorIssue]
        )
        L = ufl.inner(self.u_n, v) * ufl.dx

        # Source term: dt * f * v * dx
        source = build_source_form(
            v,  # type: ignore[reportArgumentType]
            self.msh,
            config.source_terms["u"],
            config.parameters,
        )
        if source is not None:
            L = L + dt_c * source  # type: ignore[reportOperatorIssue]

        # Natural BCs (Neumann/Robin): scaled by dt for implicit Euler
        a_bc, L_bc = build_natural_bc_forms(
            u,  # type: ignore[reportArgumentType]
            v,  # type: ignore[reportArgumentType]
            domain_geom,
            config.boundary_conditions["u"],
            config.parameters,
        )
        if a_bc is not None:
            a = a + dt_c * a_bc  # type: ignore[reportOperatorIssue]
        if L_bc is not None:
            L = L + dt_c * L_bc  # type: ignore[reportOperatorIssue]

        # Dirichlet BCs
        bcs = apply_dirichlet_bcs(
            self.V, domain_geom, config.boundary_conditions["u"], config.parameters
        )

        self.problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="plm_heat_",
            petsc_options=self._solver_options,
        )

    def step(self, t: float, dt: float) -> None:
        self.uh = self.problem.solve()
        self.u_n.x.array[:] = self.uh.x.array  # type: ignore[reportAttributeAccessIssue]

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n}  # type: ignore[reportReturnType]

    def get_num_dofs(self) -> int:
        return self.V.dofmap.index_map.size_global
