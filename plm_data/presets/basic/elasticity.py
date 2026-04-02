"""Transient small-strain linear elasticity preset."""

import numpy as np
import ufl
from dolfinx import default_real_type, fem

from plm_data.core.boundary_conditions import (
    apply_vector_dirichlet_bcs,
    build_vector_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_vector_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_BLOCK_DIRECT
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_vector_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_BETA = 0.25
_GAMMA = 0.5

_ELASTICITY_BOUNDARY_OPERATORS = {
    "dirichlet": VECTOR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": VECTOR_STANDARD_BOUNDARY_OPERATORS["neumann"],
}

_ELASTICITY_SPEC = PresetSpec(
    name="elasticity",
    category="basic",
    description=(
        "Transient small-strain isotropic linear elasticity with Rayleigh "
        "damping, vector displacement dynamics, and a derived von Mises stress."
    ),
    equations={
        "displacement": "density * d²u/dt² + C(du/dt) - div(σ(u)) = f",
        "velocity": "v = du/dt",
    },
    parameters=[
        PDEParameter("young_modulus", "Young's modulus"),
        PDEParameter("poisson_ratio", "Poisson ratio"),
        PDEParameter("density", "Mass density"),
        PDEParameter("eta_mass", "Rayleigh mass-damping coefficient"),
        PDEParameter("eta_stiffness", "Rayleigh stiffness-damping coefficient"),
    ],
    inputs={
        "displacement": InputSpec(
            name="displacement",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "forcing": InputSpec(
            name="forcing",
            shape="vector",
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "displacement": BoundaryFieldSpec(
            name="displacement",
            shape="vector",
            operators=_ELASTICITY_BOUNDARY_OPERATORS,
            description=(
                "Displacement boundary conditions. Use Dirichlet for clamped/"
                "prescribed motion and Neumann for traction-free or imposed traction "
                "boundaries."
            ),
        )
    },
    states={
        "displacement": StateSpec(name="displacement", shape="vector"),
        "velocity": StateSpec(name="velocity", shape="vector"),
    },
    outputs={
        "displacement": OutputSpec(
            name="displacement",
            shape="vector",
            output_mode="components",
            source_name="displacement",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "von_mises": OutputSpec(
            name="von_mises",
            shape="scalar",
            output_mode="scalar",
            source_name="von_mises",
            source_kind="derived",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _constrained_dofs_from_bcs(bcs: list[fem.DirichletBC]) -> np.ndarray:
    constrained_dofs: list[np.ndarray] = []
    for bc in bcs:
        dofs, _ = bc.dof_indices()
        constrained_dofs.append(dofs)
    if not constrained_dofs:
        return np.empty(0, dtype=np.int32)
    return np.unique(np.concatenate(constrained_dofs))


def _apply_dirichlet_values(
    func: fem.Function,
    bcs: list[fem.DirichletBC],
) -> None:
    for bc in bcs:
        bc.set(func.x.array)
    func.x.scatter_forward()


def _zero_dirichlet_bcs(
    V: fem.FunctionSpace,
    bcs: list[fem.DirichletBC],
) -> list[fem.DirichletBC]:
    zero = fem.Function(V)
    zero.x.array[:] = 0.0
    zero.x.scatter_forward()

    zero_bcs: list[fem.DirichletBC] = []
    for bc in bcs:
        dofs, _ = bc.dof_indices()
        zero_bcs.append(fem.dirichletbc(zero, dofs))
    return zero_bcs


def _strain(u):
    return ufl.sym(ufl.grad(u))


def _stress(u, gdim: int, lmbda, mu):
    eps = _strain(u)
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(gdim)


def _von_mises_stress(u, gdim: int, lmbda, mu):
    sigma = _stress(u, gdim, lmbda, mu)
    if gdim == 2:
        sigma_zz = lmbda * ufl.tr(_strain(u))
        sigma_3d = ufl.as_matrix(
            (
                (sigma[0, 0], sigma[0, 1], 0.0),
                (sigma[1, 0], sigma[1, 1], 0.0),
                (0.0, 0.0, sigma_zz),
            )
        )
    else:
        sigma_3d = sigma

    sigma_dev = sigma_3d - (1.0 / 3.0) * ufl.tr(sigma_3d) * ufl.Identity(3)
    return ufl.sqrt(1.5 * ufl.inner(sigma_dev, sigma_dev))


class _ElasticityProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_BLOCK_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        boundary_field = self.config.boundary_field("displacement")
        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="displacement",
            boundary_field=boundary_field,
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "neumann"},
        )
        if not any(
            entry.type == "dirichlet"
            for entries in boundary_field.sides.values()
            for entry in entries
        ):
            raise ValueError(
                "Elasticity requires at least one Dirichlet boundary condition "
                "to remove rigid-body modes."
            )

    def setup(self) -> None:
        young_modulus = self.config.parameters["young_modulus"]
        poisson_ratio = self.config.parameters["poisson_ratio"]
        density = self.config.parameters["density"]
        eta_mass = self.config.parameters["eta_mass"]
        eta_stiffness = self.config.parameters["eta_stiffness"]

        if young_modulus <= 0.0:
            raise ValueError("Elasticity parameter 'young_modulus' must be positive.")
        if not -1.0 < poisson_ratio < 0.5:
            raise ValueError(
                "Elasticity parameter 'poisson_ratio' must lie in (-1, 0.5)."
            )
        if density <= 0.0:
            raise ValueError("Elasticity parameter 'density' must be positive.")
        if eta_mass < 0.0:
            raise ValueError("Elasticity parameter 'eta_mass' must be nonnegative.")
        if eta_stiffness < 0.0:
            raise ValueError(
                "Elasticity parameter 'eta_stiffness' must be nonnegative."
            )

        domain_geom = self.load_domain_geometry()
        boundary_field = self.config.boundary_field("displacement")
        self.msh = domain_geom.mesh
        gdim = self.msh.geometry.dim

        V = fem.functionspace(self.msh, ("Lagrange", 1, (gdim,)))
        self._space_dofs = _space_num_dofs(V)

        bcs = apply_vector_dirichlet_bcs(
            V,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        zero_bcs = _zero_dirichlet_bcs(V, bcs)
        self._constrained_dofs = _constrained_dofs_from_bcs(bcs)

        self.u_n = fem.Function(V, name="displacement")
        self.v_n = fem.Function(V, name="velocity")
        self.a_n = fem.Function(V, name="acceleration")
        self.u_next = fem.Function(V, name="displacement_next")

        displacement_input = self.config.input("displacement")
        velocity_input = self.config.input("velocity")
        forcing_input = self.config.input("forcing")

        assert displacement_input.initial_condition is not None
        apply_vector_ic(
            self.u_n,
            displacement_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
            stream_id="displacement",
        )
        _apply_dirichlet_values(self.u_n, bcs)

        assert velocity_input.initial_condition is not None
        apply_vector_ic(
            self.v_n,
            velocity_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
            stream_id="velocity",
        )
        _apply_dirichlet_values(self.v_n, zero_bcs)

        dt = self.config.time.dt
        self._dt = dt
        self._a0 = 1.0 / (_BETA * dt**2)
        self._a1 = _GAMMA / (_BETA * dt)
        self._a2 = 1.0 / (_BETA * dt)
        self._a3 = 1.0 / (2.0 * _BETA) - 1.0

        lmbda = fem.Constant(
            self.msh,
            default_real_type(
                young_modulus
                * poisson_ratio
                / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
            ),
        )
        mu = fem.Constant(
            self.msh,
            default_real_type(young_modulus / (2.0 * (1.0 + poisson_ratio))),
        )
        density_c = fem.Constant(self.msh, default_real_type(density))
        eta_mass_c = fem.Constant(self.msh, default_real_type(eta_mass))
        eta_stiffness_c = fem.Constant(self.msh, default_real_type(eta_stiffness))

        def mass_form(field, test):
            return density_c * ufl.inner(field, test) * ufl.dx

        def stiffness_form(field, test):
            return ufl.inner(_stress(field, gdim, lmbda, mu), _strain(test)) * ufl.dx

        def damping_form(field, test):
            return (
                eta_mass_c * density_c * ufl.inner(field, test) * ufl.dx
                + eta_stiffness_c
                * ufl.inner(_stress(field, gdim, lmbda, mu), _strain(test))
                * ufl.dx
            )

        u = ufl.TrialFunction(V)
        w = ufl.TestFunction(V)

        zero_vec = fem.Constant(self.msh, np.zeros(gdim, dtype=default_real_type))
        external_load = ufl.inner(zero_vec, w) * ufl.dx
        assert forcing_input.source is not None
        source_form = build_vector_source_form(
            w,
            self.msh,
            forcing_input.source,
            self.config.parameters,
        )
        if source_form is not None:
            external_load = external_load + source_form

        traction_form = build_vector_natural_bc_forms(
            w,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        if traction_form is not None:
            external_load = external_load + traction_form

        # Start from zero acceleration and let the first implicit Newmark step
        # correct the state. This avoids a separate startup solve, which has proven
        # unstable under the current DOLFINx/PETSc test environment.
        self.a_n.x.array[:] = 0.0
        _apply_dirichlet_values(self.a_n, zero_bcs)

        effective_mass_predictor = (
            self._a0 * self.u_n + self._a2 * self.v_n + self._a3 * self.a_n
        )
        effective_damping_predictor = (
            self._a1 * self.u_n
            + (_GAMMA / _BETA - 1.0) * self.v_n
            + dt * (_GAMMA / (2.0 * _BETA) - 1.0) * self.a_n
        )

        effective_lhs = (
            mass_form(self._a0 * u, w)
            + damping_form(self._a1 * u, w)
            + stiffness_form(u, w)
        )
        effective_rhs = (
            external_load
            + mass_form(effective_mass_predictor, w)
            + damping_form(effective_damping_predictor, w)
        )

        self.problem = self.create_linear_problem(
            effective_lhs,
            effective_rhs,
            u=self.u_next,
            bcs=bcs,
            petsc_options_prefix="plm_elasticity_",
        )

        von_mises_space = fem.functionspace(self.msh, ("Discontinuous Lagrange", 0))
        self.von_mises_out = fem.Function(von_mises_space, name="von_mises")
        self._von_mises_expr = fem.Expression(
            _von_mises_stress(self.u_n, gdim, lmbda, mu),
            von_mises_space.element.interpolation_points,
        )

    def _update_derived_outputs(self) -> None:
        self.von_mises_out.interpolate(self._von_mises_expr)

    def step(self, t: float, dt: float) -> bool:
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason() > 0
        if not converged:
            return False

        u_prev = self.u_n.x.array.copy()
        v_prev = self.v_n.x.array.copy()
        a_prev = self.a_n.x.array.copy()
        u_next = self.u_next.x.array

        a_next = self._a0 * (u_next - u_prev) - self._a2 * v_prev - self._a3 * a_prev
        v_next = v_prev + self._dt * ((1.0 - _GAMMA) * a_prev + _GAMMA * a_next)
        if self._constrained_dofs.size:
            v_next[self._constrained_dofs] = 0.0
            a_next[self._constrained_dofs] = 0.0

        self.u_n.x.array[:] = u_next
        self.v_n.x.array[:] = v_next
        self.a_n.x.array[:] = a_next
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()
        self.a_n.x.scatter_forward()
        return True

    def get_output_fields(self) -> dict[str, fem.Function]:
        self._update_derived_outputs()
        return {
            "displacement": self.u_n,
            "velocity": self.v_n,
            "von_mises": self.von_mises_out,
        }

    def get_num_dofs(self) -> int:
        return 2 * self._space_dofs


@register_preset("elasticity")
class ElasticityPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _ELASTICITY_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _ElasticityProblem(self.spec, config)
