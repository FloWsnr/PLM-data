"""Wave equation preset: d2u/dt2 + damping * du/dt = div(c_sq * grad(u)) + f."""

import numpy as np
import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.core.source_terms import build_source_form
from plm_data.core.spatial_fields import build_ufl_field, scalar_expression_to_config
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_BETA = 0.25
_GAMMA = 0.5

_WAVE_SPEC = PresetSpec(
    name="wave",
    category="basic",
    description="Damped wave equation with scalar displacement and velocity outputs.",
    equations={
        "u": "∂²u/∂t² + damping * ∂u/∂t = ∇·(c² ∇u) + f",
        "v": "v = ∂u/∂t",
    },
    parameters=[PDEParameter("damping", "Linear damping coefficient")],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "v": InputSpec(
            name="v",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "forcing": InputSpec(
            name="forcing",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the scalar displacement field.",
        )
    },
    states={
        "u": StateSpec(name="u", shape="scalar"),
        "v": StateSpec(name="v", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
        "v": OutputSpec(
            name="v",
            shape="scalar",
            output_mode="scalar",
            source_name="v",
        ),
    },
    static_fields=[],
    supported_dimensions=[2, 3],
    coefficients={
        "c_sq": CoefficientSpec(
            name="c_sq",
            shape="scalar",
            description="Wave-speed-squared coefficient field.",
        )
    },
)


def _zero_dirichlet_bcs(
    msh,
    V: fem.FunctionSpace,
    bcs: list[fem.DirichletBC],
) -> list[fem.DirichletBC]:
    """Return zero-valued BCs on the same constrained scalar DOFs."""
    zero = fem.Constant(msh, np.float64(0.0))
    zero_bcs: list[fem.DirichletBC] = []
    for bc in bcs:
        dofs, _ = bc.dof_indices()
        zero_bcs.append(fem.dirichletbc(zero, dofs, V))
    return zero_bcs


class _WaveProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        displacement_input = self.config.input("u")
        velocity_input = self.config.input("v")
        forcing_input = self.config.input("forcing")
        c_sq_config = self.config.coefficient("c_sq")
        boundary_field = self.config.boundary_field("u")
        dt = self.config.time.dt

        bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(V, domain_geom, boundary_field, bcs)
        self.V = V if mpc is None else mpc.function_space
        self._space_dofs = (
            self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        )

        self.u_n = fem.Function(self.V, name="u")
        self.v_n = fem.Function(self.V, name="v")
        self.a_n = fem.Function(self.V, name="a")
        self.u_next = fem.Function(self.V, name="u_next")

        assert displacement_input.initial_condition is not None
        apply_ic(
            self.u_n,
            displacement_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        assert velocity_input.initial_condition is not None
        apply_ic(
            self.v_n,
            velocity_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()

        damping = self.config.parameters["damping"]
        damping_c = fem.Constant(self.msh, np.float64(damping))
        self._a0 = 1.0 / (_BETA * dt**2)
        self._a1 = _GAMMA / (_BETA * dt)
        self._a2 = 1.0 / (_BETA * dt)
        self._a3 = 1.0 / (2.0 * _BETA) - 1.0
        self._a4 = _GAMMA / _BETA - 1.0
        self._a5 = dt * (_GAMMA / (2.0 * _BETA) - 1.0)
        self._dt = dt

        c_sq = build_ufl_field(
            self.msh,
            scalar_expression_to_config(c_sq_config),
            self.config.parameters,
        )
        if c_sq is None:
            raise ValueError("Wave coefficient 'c_sq' cannot use a custom expression")

        u = ufl.TrialFunction(V)
        w = ufl.TestFunction(V)
        zero = fem.Constant(self.msh, np.float64(0.0))

        stiffness = ufl.inner(c_sq * ufl.grad(u), ufl.grad(w)) * ufl.dx
        robin_bilinear, boundary_load = build_natural_bc_forms(
            u,
            w,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        if robin_bilinear is not None:
            stiffness = stiffness + robin_bilinear

        stiffness_u_n = ufl.inner(c_sq * ufl.grad(self.u_n), ufl.grad(w)) * ufl.dx
        robin_u_n, _ = build_natural_bc_forms(
            self.u_n,
            w,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        if robin_u_n is not None:
            stiffness_u_n = stiffness_u_n + robin_u_n

        assert forcing_input.source is not None
        forcing = build_source_form(
            w,
            self.msh,
            forcing_input.source,
            self.config.parameters,
        )
        if forcing is None:
            forcing = ufl.inner(zero, w) * ufl.dx
        if boundary_load is not None:
            forcing = forcing + boundary_load

        zero_bcs = _zero_dirichlet_bcs(self.msh, V, bcs)
        startup_rhs = (
            forcing - ufl.inner(damping_c * self.v_n, w) * ufl.dx - stiffness_u_n
        )
        startup_problem = self.create_linear_problem(
            ufl.inner(u, w) * ufl.dx,
            startup_rhs,
            u=self.a_n,
            bcs=zero_bcs,
            petsc_options_prefix="plm_wave_startup_",
            mpc=mpc,
        )
        startup_problem.solve()
        startup_reason = startup_problem.solver.getConvergedReason()
        if startup_reason <= 0:
            raise RuntimeError(
                f"Wave startup solve did not converge (KSP reason={startup_reason})"
            )
        self.a_n.x.scatter_forward()

        effective_predictor = (
            self._a0 * self.u_n
            + self._a2 * self.v_n
            + self._a3 * self.a_n
            + damping_c
            * (self._a1 * self.u_n + self._a4 * self.v_n + self._a5 * self.a_n)
        )
        effective_rhs = forcing + ufl.inner(effective_predictor, w) * ufl.dx
        effective_lhs = (self._a0 + damping * self._a1) * ufl.inner(
            u, w
        ) * ufl.dx + stiffness

        self.problem = self.create_linear_problem(
            effective_lhs,
            effective_rhs,
            u=self.u_next,
            bcs=bcs,
            petsc_options_prefix="plm_wave_",
            mpc=mpc,
        )

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

        self.u_n.x.array[:] = u_next
        self.v_n.x.array[:] = v_next
        self.a_n.x.array[:] = a_next
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()
        self.a_n.x.scatter_forward()
        return True

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n, "v": self.v_n}

    def get_num_dofs(self) -> int:
        return 2 * self._space_dofs


@register_preset("wave")
class WavePreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _WAVE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _WaveProblem(self.spec, config)
