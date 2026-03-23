"""Incompressible Navier-Stokes preset using divergence-conforming DG method.

Uses Raviart-Thomas elements for velocity and DG elements for pressure,
with linearized convection and upwinding. Follows the DOLFINx Navier-Stokes demo.
"""

import numpy as np
import ufl
from dolfinx import default_real_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.config import SimulationConfig
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_domain
from plm_data.presets import register_preset
from plm_data.presets.base import TimeDependentPreset
from plm_data.presets.metadata import PDEMetadata, PDEParameter

from mpi4py import MPI


def _domain_average(msh, v):
    """Compute the average of a function over the domain."""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(
            fem.form(fem.Constant(msh, default_real_type(1.0)) * ufl.dx)
        ),
        op=MPI.SUM,
    )
    return (1 / vol) * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * ufl.dx)), op=MPI.SUM
    )


def _jump(phi, n):
    """Compute the jump of phi across facets: phi+ x n+ + phi- x n-."""
    return ufl.outer(phi("+"), n("+")) + ufl.outer(phi("-"), n("-"))


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
            field_names=["velocity_x", "velocity_y", "pressure"],
            steady_state=False,
            supported_dimensions=[2],
        )

    def setup(self, config: SimulationConfig) -> None:
        domain_geom = create_domain(config.domain)
        self.msh = domain_geom.mesh
        gdim = self.msh.geometry.dim

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
        u, p = ufl.TrialFunctions(VQ)
        v, q = ufl.TestFunctions(VQ)

        # Constants
        delta_t = fem.Constant(self.msh, default_real_type(dt))
        alpha = fem.Constant(self.msh, default_real_type(6.0 * k**2))

        h = ufl.CellDiameter(self.msh)
        n = ufl.FacetNormal(self.msh)

        # --- Viscous DG bilinear form (symmetric interior penalty) ---
        a_visc = (1.0 / Re) * (
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(ufl.avg(ufl.grad(u)), _jump(v, n)) * ufl.dS
            - ufl.inner(_jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
            + (alpha / ufl.avg(h)) * ufl.inner(_jump(u, n), _jump(v, n)) * ufl.dS
            - ufl.inner(ufl.grad(u), ufl.outer(v, n)) * ufl.ds
            - ufl.inner(ufl.outer(u, n), ufl.grad(v)) * ufl.ds
            + (alpha / h) * ufl.inner(ufl.outer(u, n), ufl.outer(v, n)) * ufl.ds
        )

        # Pressure-velocity coupling
        a_pv = -ufl.inner(p, ufl.div(v)) * ufl.dx
        a_up = -ufl.inner(ufl.div(u), q) * ufl.dx

        a_stokes = a_visc + a_pv + a_up

        # --- Boundary conditions ---
        # Parse velocity BCs from config: each boundary gets a vector value
        bc_configs = config.boundary_conditions.get("velocity", {})
        bc_values = {}
        for bnd_name, bc_conf in bc_configs.items():
            if bc_conf.type == "dirichlet":
                val = bc_conf.value
                if isinstance(val, list):
                    bc_values[bnd_name] = val
                else:
                    bc_values[bnd_name] = [float(val), 0.0]

        # u_D on DG Lagrange vector space: stores full boundary velocity
        # (RT elements only capture normal flux, losing tangential components)
        u_D = fem.Function(self._W)
        self._interpolate_boundary_velocity(u_D, domain_geom, bc_values)

        # u_D_rt on RT space for strong Dirichlet BC (normal component)
        u_D_rt = fem.Function(V)
        u_D_rt.interpolate(u_D)

        # Strong Dirichlet BC on RT space (constrains normal velocity)
        self.msh.topology.create_connectivity(
            self.msh.topology.dim - 1, self.msh.topology.dim
        )
        boundary_facets = dmesh.exterior_facet_indices(self.msh.topology)
        boundary_vel_dofs = fem.locate_dofs_topological(
            V, self.msh.topology.dim - 1, boundary_facets
        )
        bc_u = fem.dirichletbc(u_D_rt, boundary_vel_dofs)
        bcs = [bc_u]

        # Weak BC terms: use u_D (DG Lagrange) for full velocity enforcement
        L_bc = (1.0 / Re) * (
            -ufl.inner(ufl.outer(u_D, n), ufl.grad(v)) * ufl.ds
            + (alpha / h) * ufl.inner(ufl.outer(u_D, n), ufl.outer(v, n)) * ufl.ds
        )

        # Zero RHS for pressure equation (needed for block structure)
        L_bc += ufl.inner(fem.Constant(self.msh, default_real_type(0.0)), q) * ufl.dx

        # --- Body force (source term) ---
        source_config = config.source_terms.get("velocity")
        if source_config and source_config.type == "custom":
            force_x = config.parameters["force_x"]
            force_y = config.parameters["force_y"]
            f_body = ufl.as_vector(
                [default_real_type(force_x), default_real_type(force_y)]
            )
            L_bc = L_bc + ufl.inner(f_body, v) * ufl.dx

        # --- Prepare output sub-spaces (needed for IC and output) ---
        self._u_vis = fem.Function(self._W, name="u_vis")
        W_x, self._dofs_x = self._W.sub(0).collapse()
        W_y, self._dofs_y = self._W.sub(1).collapse()
        self._u_x = fem.Function(W_x, name="velocity_x")
        self._u_y = fem.Function(W_y, name="velocity_y")

        # --- Solution functions ---
        self.u_h = fem.Function(V)
        self.p_h = fem.Function(Q)
        self.p_h.name = "p"

        # --- Initial condition ---
        ic_configs = config.initial_conditions
        use_stokes_ic = (
            "velocity" in ic_configs and ic_configs["velocity"].type == "custom"
        )

        if use_stokes_ic or not ic_configs:
            # Custom IC: Stokes solve for divergence-free initial velocity
            stokes_problem = LinearProblem(
                ufl.extract_blocks(a_stokes),
                ufl.extract_blocks(L_bc),
                u=[self.u_h, self.p_h],
                bcs=bcs,
                kind="mpi",
                petsc_options_prefix="plm_ns_stokes_",
                petsc_options=self._solver_options,
            )
            stokes_problem.solve()
            self.p_h.x.array[:] -= _domain_average(self.msh, self.p_h)
        else:
            # Per-component ICs: apply standard IC types to velocity_x, velocity_y
            self._apply_component_ics(ic_configs, V, config.seed)

        # Store previous velocity
        self.u_n = fem.Function(V, name="u_prev")
        self.u_n.x.array[:] = self.u_h.x.array

        # --- Add time-stepping and convective terms ---
        # Upwind flux
        lmbda = ufl.conditional(ufl.gt(ufl.dot(self.u_n, n), 0), 1, 0)
        u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")

        a_ns = a_stokes + (
            ufl.inner(u / delta_t, v) * ufl.dx
            - ufl.inner(u, ufl.div(ufl.outer(v, self.u_n))) * ufl.dx
            + ufl.inner((ufl.dot(self.u_n, n))("+") * u_uw, v("+")) * ufl.dS
            + ufl.inner((ufl.dot(self.u_n, n))("-") * u_uw, v("-")) * ufl.dS
            + ufl.inner(ufl.dot(self.u_n, n) * lmbda * u, v) * ufl.ds
        )

        L_ns = L_bc + (
            ufl.inner(self.u_n / delta_t, v) * ufl.dx
            - ufl.inner(ufl.dot(self.u_n, n) * (1 - lmbda) * u_D, v) * ufl.ds
        )

        self._ns_problem = LinearProblem(
            ufl.extract_blocks(a_ns),
            ufl.extract_blocks(L_ns),
            u=[self.u_h, self.p_h],
            bcs=bcs,
            kind="mpi",
            petsc_options_prefix="plm_ns_",
            petsc_options=self._solver_options,
        )

        self._V = V
        self._Q = Q

    def _interpolate_boundary_velocity(self, u_D, domain_geom, bc_values):
        """Interpolate velocity boundary condition from per-boundary config."""
        # Default: zero velocity everywhere
        u_D.x.array[:] = 0.0

        if not bc_values:
            return

        # Validate boundary names
        for bnd_name in bc_values:
            if bnd_name not in domain_geom.boundary_names:
                raise ValueError(
                    f"Unknown boundary '{bnd_name}'. "
                    f"Available: {list(domain_geom.boundary_names.keys())}"
                )

        # Interpolate combined BC expression for all boundaries
        def _bc_expr(x):
            result = np.zeros((self.msh.geometry.dim, x.shape[1]))
            coords = self.msh.geometry.x
            bounds = tuple(
                (float(coords[:, d].min()), float(coords[:, d].max()))
                for d in range(self.msh.geometry.dim)
            )
            tol = 1e-10

            for bnd_name, value in bc_values.items():
                # Determine which points are on this boundary
                if bnd_name == "x-":
                    mask = np.abs(x[0] - bounds[0][0]) < tol
                elif bnd_name == "x+":
                    mask = np.abs(x[0] - bounds[0][1]) < tol
                elif bnd_name == "y-":
                    mask = np.abs(x[1] - bounds[1][0]) < tol
                elif bnd_name == "y+":
                    mask = np.abs(x[1] - bounds[1][1]) < tol
                else:
                    continue

                for d in range(min(len(value), self.msh.geometry.dim)):
                    result[d, mask] = value[d]

            return result

        u_D.interpolate(_bc_expr)

    def _apply_component_ics(self, ic_configs, V, seed):
        """Apply per-component initial conditions to velocity.

        Uses velocity_x and velocity_y IC configs to set scalar components
        on the DG Lagrange vector space, then interpolates into RT space.
        """
        if "velocity_x" in ic_configs:
            apply_ic(self._u_x, ic_configs["velocity_x"], seed=seed)
        if "velocity_y" in ic_configs:
            apply_ic(self._u_y, ic_configs["velocity_y"], seed=seed)

        # Write scalar components into the DG Lagrange vector function
        self._u_vis.x.array[self._dofs_x] = self._u_x.x.array
        self._u_vis.x.array[self._dofs_y] = self._u_y.x.array

        # Interpolate from DG Lagrange vector into RT space
        self.u_h.interpolate(self._u_vis)

    def step(self, t: float, dt: float) -> None:
        self._ns_problem.solve()
        self.p_h.x.array[:] -= _domain_average(self.msh, self.p_h)
        self.u_n.x.array[:] = self.u_h.x.array

    def get_output_fields(self) -> dict[str, fem.Function]:
        # Interpolate RT velocity into DG Lagrange vector space
        self._u_vis.interpolate(self.u_h)

        # Extract scalar components
        self._u_x.x.array[:] = self._u_vis.x.array[self._dofs_x]
        self._u_y.x.array[:] = self._u_vis.x.array[self._dofs_y]

        return {
            "velocity_x": self._u_x,
            "velocity_y": self._u_y,
            "pressure": self.p_h,
        }

    def get_num_dofs(self) -> int:
        return (
            self._V.dofmap.index_map.size_global + self._Q.dofmap.index_map.size_global
        )
