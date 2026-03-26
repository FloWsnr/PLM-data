"""Incompressible Navier-Stokes preset using a divergence-conforming DG method."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import default_real_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from plm_data.core.initial_conditions import apply_vector_ic
from plm_data.core.fem_utils import dg_jump, domain_average
from plm_data.core.mesh import create_domain
from plm_data.core.source_terms import build_vector_source_form
from plm_data.core.spatial_fields import (
    build_vector_interpolator,
    component_labels_for_dim,
)
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.metadata import FieldSpec, PDEParameter, PresetSpec


_NAVIER_STOKES_SPEC = PresetSpec(
    name="navier_stokes",
    category="fluids",
    description=(
        "Incompressible Navier-Stokes equations using a divergence-conforming DG "
        "method with Raviart-Thomas velocity and DG pressure."
    ),
    equations={
        "velocity": "du/dt + (u.grad)u = -grad(p) + (1/Re)*laplacian(u)",
        "pressure": "div(u) = 0",
    },
    parameters=[
        PDEParameter("Re", "Reynolds number"),
        PDEParameter("k", "Polynomial degree for velocity and pressure spaces"),
    ],
    fields={
        "velocity": FieldSpec(
            name="velocity",
            shape="vector",
            allow_boundary_conditions=True,
            allow_source=True,
            allow_initial_condition=True,
            output_mode="components",
        ),
        "pressure": FieldSpec(
            name="pressure",
            shape="scalar",
            allow_boundary_conditions=False,
            allow_source=False,
            allow_initial_condition=False,
            output_mode="scalar",
        ),
    },
    family="custom",
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _NavierStokesProblem(TransientLinearProblem):
    def setup(self) -> None:
        domain_geom = create_domain(self.config.domain)
        self.domain_geom = domain_geom
        self.msh = domain_geom.mesh
        self.gdim = self.msh.geometry.dim
        self._component_labels = component_labels_for_dim(self.gdim)

        Re = self.config.parameters["Re"]
        k = int(self.config.parameters["k"])
        dt = self.config.time.dt

        V = fem.functionspace(self.msh, ("Raviart-Thomas", k + 1))
        Q = fem.functionspace(self.msh, ("Discontinuous Lagrange", k))
        VQ = ufl.MixedFunctionSpace(V, Q)
        self._W = fem.functionspace(
            self.msh, ("Discontinuous Lagrange", k + 1, (self.gdim,))
        )

        u, p = ufl.TrialFunctions(VQ)
        v, q = ufl.TestFunctions(VQ)

        delta_t = fem.Constant(self.msh, default_real_type(dt))
        alpha = fem.Constant(self.msh, default_real_type(6.0 * k**2))
        h = ufl.CellDiameter(self.msh)
        n = ufl.FacetNormal(self.msh)

        a_visc = (1.0 / Re) * (
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            - ufl.inner(ufl.avg(ufl.grad(u)), dg_jump(v, n)) * ufl.dS
            - ufl.inner(dg_jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
            + (alpha / ufl.avg(h)) * ufl.inner(dg_jump(u, n), dg_jump(v, n)) * ufl.dS
            - ufl.inner(ufl.grad(u), ufl.outer(v, n)) * ufl.ds
            - ufl.inner(ufl.outer(u, n), ufl.grad(v)) * ufl.ds
            + (alpha / h) * ufl.inner(ufl.outer(u, n), ufl.outer(v, n)) * ufl.ds
        )
        a_pv = -ufl.inner(p, ufl.div(v)) * ufl.dx
        a_up = -ufl.inner(ufl.div(u), q) * ufl.dx
        a_stokes = a_visc + a_pv + a_up

        u_D = fem.Function(self._W, name="velocity_bc")
        self._interpolate_boundary_velocity(u_D)

        u_D_rt = fem.Function(V)
        u_D_rt.interpolate(u_D)

        self.msh.topology.create_connectivity(
            self.msh.topology.dim - 1, self.msh.topology.dim
        )
        boundary_facets = dmesh.exterior_facet_indices(self.msh.topology)
        boundary_vel_dofs = fem.locate_dofs_topological(
            V,
            self.msh.topology.dim - 1,
            boundary_facets,
        )
        bc_u = fem.dirichletbc(u_D_rt, boundary_vel_dofs)
        bcs = [bc_u]

        L_bc = (1.0 / Re) * (
            -ufl.inner(ufl.outer(u_D, n), ufl.grad(v)) * ufl.ds
            + (alpha / h) * ufl.inner(ufl.outer(u_D, n), ufl.outer(v, n)) * ufl.ds
        )
        L_bc += ufl.inner(fem.Constant(self.msh, default_real_type(0.0)), q) * ufl.dx

        velocity_field = self.config.field("velocity")
        assert velocity_field.source is not None
        source_form = build_vector_source_form(
            v,
            self.msh,
            velocity_field.source,
            self.config.parameters,
        )
        if source_form is not None:
            L_bc = L_bc + source_form

        self._u_vis = fem.Function(self._W, name="velocity")
        self.u_h = fem.Function(V, name="velocity_rt")
        self.p_h = fem.Function(Q, name="pressure")

        assert velocity_field.initial_condition is not None
        if velocity_field.initial_condition.type == "custom":
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
            self.p_h.x.array[:] -= domain_average(self.msh, self.p_h)
        else:
            apply_vector_ic(
                self._u_vis,
                velocity_field.initial_condition,
                self.config.parameters,
                seed=self.config.seed,
            )
            self.u_h.interpolate(self._u_vis)

        self.u_n = fem.Function(V, name="u_prev")
        self.u_n.x.array[:] = self.u_h.x.array

        lmbda = ufl.conditional(ufl.gt(ufl.dot(self.u_n, n), 0), 1, 0)
        u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")

        a_ns = a_stokes + (
            ufl.inner(u / delta_t, v) * ufl.dx
            - ufl.inner(u, ufl.div(ufl.outer(v, self.u_n))) * ufl.dx
            + ufl.inner((ufl.dot(self.u_n, n))("+") * u_uw, v("+")) * ufl.dS  # type: ignore[reportCallIssue, reportOptionalCall]
            + ufl.inner((ufl.dot(self.u_n, n))("-") * u_uw, v("-")) * ufl.dS  # type: ignore[reportCallIssue, reportOptionalCall]
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

    def _interpolate_boundary_velocity(self, u_D: fem.Function) -> None:
        """Interpolate a piecewise boundary velocity field onto a DG vector space."""
        velocity_bcs = self.config.field("velocity").boundary_conditions
        for name, bc in velocity_bcs.items():
            if bc.type != "dirichlet":
                raise ValueError(
                    f"Navier-Stokes boundary '{name}' must be dirichlet, got '{bc.type}'"
                )

        coords = self.msh.geometry.x
        bounds = tuple(
            (float(coords[:, d].min()), float(coords[:, d].max()))
            for d in range(self.gdim)
        )
        tol = 1e-10
        axis_map = {"x": 0, "y": 1, "z": 2}
        boundary_checks: dict[str, tuple[int, float]] = {}
        for name in self.domain_geom.boundary_names:
            axis = axis_map[name[0]]
            limit = bounds[axis][1] if name[1] == "+" else bounds[axis][0]
            boundary_checks[name] = (axis, limit)

        boundary_interpolators = {}
        for name, bc in velocity_bcs.items():
            interpolator = build_vector_interpolator(
                bc.value,
                self.gdim,
                self.config.parameters,
            )
            if interpolator is None:
                raise ValueError(f"Boundary '{name}' cannot use custom vector values")
            boundary_interpolators[name] = interpolator

        def _bc_expr(x):
            result = np.zeros((self.gdim, x.shape[1]))
            for name, interpolator in boundary_interpolators.items():
                axis, limit = boundary_checks[name]
                mask = np.abs(x[axis] - limit) < tol
                if not np.any(mask):
                    continue
                values = interpolator(x)
                result[:, mask] = values[:, mask]
            return result

        u_D.interpolate(_bc_expr)

    def step(self, t: float, dt: float) -> bool:
        self._ns_problem.solve()
        self.p_h.x.array[:] -= domain_average(self.msh, self.p_h)
        self.u_n.x.array[:] = self.u_h.x.array
        return self._ns_problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        self._u_vis.interpolate(self.u_h)
        return {"velocity": self._u_vis, "pressure": self.p_h}

    def get_num_dofs(self) -> int:
        return (
            self._V.dofmap.index_map.size_global + self._Q.dofmap.index_map.size_global
        )


@register_preset("navier_stokes")
class NavierStokesPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _NAVIER_STOKES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _NavierStokesProblem(self.spec, config)
