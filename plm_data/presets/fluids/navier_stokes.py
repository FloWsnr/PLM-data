"""Incompressible Navier-Stokes preset using divergence-conforming DG method.

Uses Raviart-Thomas elements for velocity and DG elements for pressure,
with linearized convection and upwinding. Follows the DOLFINx Navier-Stokes demo.
"""

import numpy as np
import ufl
from dolfinx import default_real_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.config import SimulationConfig
from plm_data.core.fem_utils import dg_jump, domain_average
from plm_data.core.initial_conditions import apply_vector_ic
from plm_data.core.mesh import DomainGeometry, create_domain
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import TimeDependentPreset
from plm_data.presets.metadata import PDEMetadata, PDEParameter

_COMPONENT_LABELS = ["x", "y", "z"]


@register_preset("navier_stokes")
class NavierStokesPreset(TimeDependentPreset):
    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="navier_stokes",
            category="fluids",
            description=(
                "Incompressible Navier-Stokes equations using a divergence-conforming "
                "DG method with Raviart-Thomas velocity and DG pressure. "
                "Linearized convection with upwinding ensures exact mass conservation."
            ),
            equations={
                "velocity": "du/dt + (u.grad)u = -grad(p) + (1/Re)*laplacian(u)",
                "pressure": "div(u) = 0",
            },
            parameters=[
                PDEParameter("Re", "Reynolds number"),
                PDEParameter(
                    "k", "Polynomial degree for velocity (RT_{k+1}) and pressure (DG_k)"
                ),
            ],
            field_names=["velocity_x", "velocity_y", "velocity_z", "pressure"],
            steady_state=False,
            supported_dimensions=[2, 3],
        )

    def setup(self, config: SimulationConfig) -> None:
        domain_geom = create_domain(config.domain)
        self.msh = domain_geom.mesh
        gdim = self.msh.geometry.dim

        # Build dimension-dependent velocity component names
        self._velocity_names = [f"velocity_{c}" for c in _COMPONENT_LABELS[:gdim]]

        Re = config.parameters["Re"]
        k = int(config.parameters["k"])
        dt = config.dt

        # Function spaces
        V = fem.functionspace(self.msh, ("Raviart-Thomas", k + 1))
        Q = fem.functionspace(self.msh, ("Discontinuous Lagrange", k))
        VQ = ufl.MixedFunctionSpace(V, Q)

        # Visualization/output space (DG Lagrange vector)
        self._W = fem.functionspace(
            self.msh, ("Discontinuous Lagrange", k + 1, (gdim,))
        )

        # Trial and test functions
        u, p = ufl.TrialFunctions(VQ)  # type: ignore[reportAssignmentType]
        v, q = ufl.TestFunctions(VQ)  # type: ignore[reportAssignmentType]

        # Constants
        delta_t = fem.Constant(self.msh, default_real_type(dt))
        alpha = fem.Constant(self.msh, default_real_type(6.0 * k**2))

        h = ufl.CellDiameter(self.msh)
        n = ufl.FacetNormal(self.msh)

        # --- Viscous DG bilinear form (symmetric interior penalty) ---
        a_visc = (1.0 / Re) * (  # type: ignore[reportOperatorIssue]
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # type: ignore[reportOperatorIssue]
            - ufl.inner(ufl.avg(ufl.grad(u)), dg_jump(v, n)) * ufl.dS
            - ufl.inner(dg_jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
            + (alpha / ufl.avg(h)) * ufl.inner(dg_jump(u, n), dg_jump(v, n)) * ufl.dS  # type: ignore[reportOperatorIssue]
            - ufl.inner(ufl.grad(u), ufl.outer(v, n)) * ufl.ds
            - ufl.inner(ufl.outer(u, n), ufl.grad(v)) * ufl.ds
            + (alpha / h) * ufl.inner(ufl.outer(u, n), ufl.outer(v, n)) * ufl.ds  # type: ignore[reportOperatorIssue]
        )

        # Pressure-velocity coupling
        a_pv = -ufl.inner(p, ufl.div(v)) * ufl.dx  # type: ignore[reportOptionalOperand]
        a_up = -ufl.inner(ufl.div(u), q) * ufl.dx  # type: ignore[reportOptionalOperand]

        a_stokes = a_visc + a_pv + a_up  # type: ignore[reportOperatorIssue]

        # --- Boundary conditions ---
        # Parse velocity BCs from config: each boundary gets a vector value
        bc_configs = config.boundary_conditions.get("velocity", {})
        bc_values: dict[str, list[float]] = {}
        for bnd_name, bc_conf in bc_configs.items():
            if bc_conf.type == "dirichlet":
                val = bc_conf.value
                if isinstance(val, list):
                    bc_values[bnd_name] = val
                else:
                    bc_values[bnd_name] = [float(val)] + [0.0] * (gdim - 1)  # type: ignore[reportArgumentType]

        # u_D on DG Lagrange vector space: stores full boundary velocity
        # (RT elements only capture normal flux, losing tangential components)
        u_D = fem.Function(self._W)
        self._interpolate_boundary_velocity(u_D, domain_geom, bc_values)

        # u_D_rt on RT space for strong Dirichlet BC (normal component)
        u_D_rt = fem.Function(V)
        u_D_rt.interpolate(u_D)  # type: ignore[reportAttributeAccessIssue]

        # Strong Dirichlet BC on RT space (constrains normal velocity)
        self.msh.topology.create_connectivity(
            self.msh.topology.dim - 1, self.msh.topology.dim
        )
        boundary_facets = dmesh.exterior_facet_indices(self.msh.topology)
        boundary_vel_dofs = fem.locate_dofs_topological(
            V, self.msh.topology.dim - 1, boundary_facets
        )
        bc_u = fem.dirichletbc(u_D_rt, boundary_vel_dofs)  # type: ignore[reportArgumentType]
        bcs = [bc_u]

        # Weak BC terms: use u_D (DG Lagrange) for full velocity enforcement
        L_bc = (1.0 / Re) * (  # type: ignore[reportOperatorIssue]
            -ufl.inner(ufl.outer(u_D, n), ufl.grad(v)) * ufl.ds  # type: ignore[reportOptionalOperand]
            + (alpha / h) * ufl.inner(ufl.outer(u_D, n), ufl.outer(v, n)) * ufl.ds  # type: ignore[reportOperatorIssue]
        )

        # Zero RHS for pressure equation (needed for block structure)
        L_bc += ufl.inner(fem.Constant(self.msh, default_real_type(0.0)), q) * ufl.dx  # type: ignore[reportOperatorIssue]

        # --- Body force (source term) ---
        source_form = build_vector_source_form(
            v,  # type: ignore[reportArgumentType]
            self.msh,
            self._velocity_names,
            config.source_terms,
            config.parameters,
        )
        if source_form is not None:
            L_bc = L_bc + source_form

        # --- Prepare output sub-spaces (needed for IC and output) ---
        self._u_vis = fem.Function(self._W, name="u_vis")
        self._component_funcs: list[fem.Function] = []
        self._component_dofs: list[np.ndarray] = []
        for i, name in enumerate(self._velocity_names):
            W_i, dofs_i = self._W.sub(i).collapse()
            self._component_funcs.append(fem.Function(W_i, name=name))
            self._component_dofs.append(dofs_i)

        # --- Solution functions ---
        self.u_h = fem.Function(V)
        self.p_h = fem.Function(Q)
        self.p_h.name = "p"  # type: ignore[reportAttributeAccessIssue]

        # --- Initial condition ---
        ic_configs = config.initial_conditions
        use_stokes_ic = (
            "velocity" in ic_configs and ic_configs["velocity"].type == "custom"
        )

        if use_stokes_ic or not ic_configs:
            # Custom IC: Stokes solve for divergence-free initial velocity
            stokes_problem = LinearProblem(
                ufl.extract_blocks(a_stokes),  # type: ignore[reportArgumentType]
                ufl.extract_blocks(L_bc),  # type: ignore[reportArgumentType]
                u=[self.u_h, self.p_h],  # type: ignore[reportArgumentType]
                bcs=bcs,
                kind="mpi",
                petsc_options_prefix="plm_ns_stokes_",
                petsc_options=self._solver_options,
            )
            stokes_problem.solve()
            self.p_h.x.array[:] -= domain_average(self.msh, self.p_h)  # type: ignore[reportAttributeAccessIssue]
        else:
            # Per-component ICs via shared utility
            apply_vector_ic(
                self._u_vis,  # type: ignore[reportArgumentType]
                self._component_funcs,  # type: ignore[reportArgumentType]
                self._component_dofs,
                ic_configs,
                self._velocity_names,
                config.parameters,
                seed=config.seed,
            )
            self.u_h.interpolate(self._u_vis)  # type: ignore[reportAttributeAccessIssue]

        # Store previous velocity
        self.u_n = fem.Function(V, name="u_prev")
        self.u_n.x.array[:] = self.u_h.x.array  # type: ignore[reportAttributeAccessIssue]

        # --- Add time-stepping and convective terms ---
        # Upwind flux
        lmbda = ufl.conditional(ufl.gt(ufl.dot(self.u_n, n), 0), 1, 0)  # type: ignore[reportOperatorIssue]
        u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")  # type: ignore[reportOperatorIssue]

        a_ns = a_stokes + (  # type: ignore[reportOperatorIssue]
            ufl.inner(u / delta_t, v) * ufl.dx  # type: ignore[reportOperatorIssue]
            - ufl.inner(u, ufl.div(ufl.outer(v, self.u_n))) * ufl.dx  # type: ignore[reportOperatorIssue]
            + ufl.inner((ufl.dot(self.u_n, n))("+") * u_uw, v("+")) * ufl.dS  # type: ignore[reportOptionalCall, reportOperatorIssue]
            + ufl.inner((ufl.dot(self.u_n, n))("-") * u_uw, v("-")) * ufl.dS  # type: ignore[reportOptionalCall, reportOperatorIssue]
            + ufl.inner(ufl.dot(self.u_n, n) * lmbda * u, v) * ufl.ds
        )

        L_ns = L_bc + (  # type: ignore[reportOperatorIssue]
            ufl.inner(self.u_n / delta_t, v) * ufl.dx
            - ufl.inner(ufl.dot(self.u_n, n) * (1 - lmbda) * u_D, v) * ufl.ds  # type: ignore[reportOperatorIssue]
        )

        self._ns_problem = LinearProblem(
            ufl.extract_blocks(a_ns),  # type: ignore[reportArgumentType]
            ufl.extract_blocks(L_ns),  # type: ignore[reportArgumentType]
            u=[self.u_h, self.p_h],  # type: ignore[reportArgumentType]
            bcs=bcs,
            kind="mpi",
            petsc_options_prefix="plm_ns_",
            petsc_options=self._solver_options,
        )

        self._V = V
        self._Q = Q

    def _interpolate_boundary_velocity(
        self,
        u_D: fem.Function,
        domain_geom: DomainGeometry,
        bc_values: dict[str, list[float]],
    ) -> None:
        """Interpolate velocity boundary condition from per-boundary config."""
        u_D.x.array[:] = 0.0

        if not bc_values:
            return

        for bnd_name in bc_values:
            if bnd_name not in domain_geom.boundary_names:
                raise ValueError(
                    f"Unknown boundary '{bnd_name}'. "
                    f"Available: {list(domain_geom.boundary_names.keys())}"
                )

        gdim = self.msh.geometry.dim
        coords = self.msh.geometry.x
        bounds = tuple(
            (float(coords[:, d].min()), float(coords[:, d].max())) for d in range(gdim)
        )
        tol = 1e-10

        # Build boundary name → (axis, limit) mapping from geometry
        axis_map = {"x": 0, "y": 1, "z": 2}
        boundary_checks: dict[str, tuple[int, float]] = {}
        for name in domain_geom.boundary_names:
            axis = axis_map[name[0]]
            limit = bounds[axis][1] if name[1] == "+" else bounds[axis][0]
            boundary_checks[name] = (axis, limit)

        def _bc_expr(x):
            result = np.zeros((gdim, x.shape[1]))
            for bnd_name, value in bc_values.items():
                if bnd_name not in boundary_checks:
                    continue
                axis, limit = boundary_checks[bnd_name]
                mask = np.abs(x[axis] - limit) < tol
                for d in range(min(len(value), gdim)):
                    result[d, mask] = value[d]
            return result

        u_D.interpolate(_bc_expr)

    def step(self, t: float, dt: float) -> bool:
        self._ns_problem.solve()
        self.p_h.x.array[:] -= domain_average(self.msh, self.p_h)  # type: ignore[reportAttributeAccessIssue]
        self.u_n.x.array[:] = self.u_h.x.array  # type: ignore[reportAttributeAccessIssue]
        return self._ns_problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        # Interpolate RT velocity into DG Lagrange vector space
        self._u_vis.interpolate(self.u_h)  # type: ignore[reportAttributeAccessIssue]

        # Extract scalar components
        fields: dict[str, fem.Function] = {}
        for name, func, dofs in zip(
            self._velocity_names, self._component_funcs, self._component_dofs
        ):
            func.x.array[:] = self._u_vis.x.array[dofs]  # type: ignore[reportAttributeAccessIssue]
            fields[name] = func
        fields["pressure"] = self.p_h
        return fields  # type: ignore[reportReturnType]

    def get_num_dofs(self) -> int:
        return (
            self._V.dofmap.index_map.size_global + self._Q.dofmap.index_map.size_global
        )
