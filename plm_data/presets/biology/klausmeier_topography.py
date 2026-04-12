"""Klausmeier vegetation-topography preset."""

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
)
from plm_data.core.stochastic import (
    build_scalar_coefficient,
    build_scalar_state_stochastic_term,
)
from plm_data.core.spatial_fields import build_interpolator, scalar_expression_to_config
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
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_KLAUSMEIER_TOPOGRAPHY_SPEC = PresetSpec(
    name="klausmeier_topography",
    category="biology",
    description=(
        "Klausmeier vegetation model on topography. Two-species reaction-diffusion "
        "system coupling water w and plant biomass n on a static terrain T(x,y). "
        "Water flows downhill along the topography gradient, creating vegetation "
        "patterns in valleys and along hillsides."
    ),
    equations={
        "w": "dw/dt = a - w - w*n^2 + D*laplacian(w) + V*div(w*grad(T))",
        "n": "dn/dt = w*n^2 - m*n + Dn*laplacian(n)",
    },
    parameters=[
        PDEParameter("a", "Rainfall rate"),
        PDEParameter("m", "Plant mortality rate"),
        PDEParameter("D", "Water diffusion coefficient"),
        PDEParameter("Dn", "Vegetation diffusion coefficient"),
        PDEParameter(
            "V",
            "Topography advection strength (V>0: water flows downhill, i.e. against grad(T))",
        ),
    ],
    inputs={
        "w": InputSpec(
            name="w",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "n": InputSpec(
            name="n",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "w": BoundaryFieldSpec(
            name="w",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for water w.",
        ),
        "n": BoundaryFieldSpec(
            name="n",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for vegetation biomass n.",
        ),
    },
    states={
        "w": StateSpec(
            name="w",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "n": StateSpec(
            name="n",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "w": OutputSpec(
            name="w",
            shape="scalar",
            output_mode="scalar",
            source_name="w",
        ),
        "n": OutputSpec(
            name="n",
            shape="scalar",
            output_mode="scalar",
            source_name="n",
        ),
        "topography": OutputSpec(
            name="topography",
            shape="scalar",
            output_mode="scalar",
            source_name="topography",
            source_kind="derived",
        ),
    },
    static_fields=["topography"],
    supported_dimensions=[2, 3],
    coefficients={
        "topography": CoefficientSpec(
            name="topography",
            shape="scalar",
            description=(
                "Static terrain height T(x,y). Hills have positive values, valleys "
                "negative. Water flows downhill (against grad(T)) when V>0."
            ),
            allow_randomization=True,
        ),
    },
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _KlausmeierTopographyProblem(TransientLinearProblem):
    supported_solver_strategies = (
        CONSTANT_LHS_SCALAR_SPD,
        CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    )

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="w",
            boundary_field=self.config.boundary_field("w"),
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="n",
            boundary_field=self.config.boundary_field("n"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        w_input = self.config.input("w")
        n_input = self.config.input("n")
        w_boundary_field = self.config.boundary_field("w")
        n_boundary_field = self.config.boundary_field("n")

        dt = self.config.time.dt
        a = self.config.parameters["a"]
        m = self.config.parameters["m"]
        D = self.config.parameters["D"]
        Dn = self.config.parameters["Dn"]
        V_param = self.config.parameters["V"]

        w_bcs = apply_dirichlet_bcs(
            V, domain_geom, w_boundary_field, self.config.parameters
        )
        n_bcs = apply_dirichlet_bcs(
            V, domain_geom, n_boundary_field, self.config.parameters
        )
        w_mpc = self.create_periodic_constraint(V, domain_geom, w_boundary_field, w_bcs)
        n_mpc = self.create_periodic_constraint(V, domain_geom, n_boundary_field, n_bcs)

        self.V_w = V if w_mpc is None else w_mpc.function_space
        self.V_n = V if n_mpc is None else n_mpc.function_space
        self._num_dofs = _space_num_dofs(self.V_w) + _space_num_dofs(self.V_n)

        self.w_n = fem.Function(self.V_w, name="w")
        self.n_n = fem.Function(self.V_n, name="n")
        self.w_h = fem.Function(self.V_w, name="w_next")
        self.n_h = fem.Function(self.V_n, name="n_next")

        assert w_input.initial_condition is not None
        apply_ic(
            self.w_n,
            w_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        if w_mpc is not None:
            w_mpc.backsubstitution(self.w_n)
        assert n_input.initial_condition is not None
        apply_ic(
            self.n_n,
            n_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
        )
        if n_mpc is not None:
            n_mpc.backsubstitution(self.n_n)
        self.w_n.x.scatter_forward()
        self.n_n.x.scatter_forward()

        # Build topography UFL expression for use in variational forms.
        # ufl.grad(T_ufl) gives the exact symbolic gradient.
        topo_config = scalar_expression_to_config(self.config.coefficient("topography"))
        T_ufl = build_scalar_coefficient(self, "topography")
        if T_ufl is None:
            raise ValueError(
                "Klausmeier topography coefficient cannot use a custom expression."
            )

        # Reuse the randomized coefficient directly when present so output matches
        # the field used in the forms.
        if isinstance(T_ufl, fem.Function):
            self.topography_out = T_ufl
        else:
            topo_interp = build_interpolator(topo_config, self.config.parameters)
            if topo_interp is None:
                raise ValueError(
                    "Klausmeier topography coefficient cannot use a custom expression."
                )
            self.topography_out = fem.Function(self.V_w, name="topography")
            self.topography_out.interpolate(topo_interp)
            self.topography_out.x.scatter_forward()

        w_trial = ufl.TrialFunction(V)
        w_test = ufl.TestFunction(V)
        n_trial = ufl.TrialFunction(V)
        n_test = ufl.TestFunction(V)

        # w-equation: (w_h - w_n)/dt = a - w_h - w_n*n_n^2 + D*laplacian(w_h)
        #                               + V*div(w*grad(T))
        #
        # T is the terrain HEIGHT (hills = positive, valleys = negative).
        # V > 0 means water flows DOWNHILL (against grad(T), into valleys).
        #
        # Topography term weak form (IBP, zero-flux or periodic BC):
        #   +V*div(w*grad(T)) * phi → -V * w * grad(T)·grad(phi)
        #
        # Two discretisation options:
        #   SPD (explicit advection): advection on RHS using w_n.
        #     LHS is constant and SPD: mass + dt*(decay_mass + D*stiffness)
        #     Stable only for small V (CFL: dt*V*|grad(T)|/h < 1).
        #   NONSYMMETRIC (implicit advection): advection on LHS using w_trial.
        #     LHS is constant but non-symmetric: same SPD base + V*advection.
        #     Unconditionally stable for any V; required for large advection.
        implicit_advection = (
            self.config.solver.strategy == CONSTANT_LHS_SCALAR_NONSYMMETRIC
        )

        lhs_w = (
            ufl.inner(w_trial, w_test) * ufl.dx
            + dt * ufl.inner(w_trial, w_test) * ufl.dx
            + dt * D * ufl.inner(ufl.grad(w_trial), ufl.grad(w_test)) * ufl.dx
        )
        rhs_w = (
            ufl.inner(self.w_n, w_test) * ufl.dx
            + dt * a * w_test * ufl.dx
            - dt * ufl.inner(self.w_n * self.n_n**2, w_test) * ufl.dx
        )
        if implicit_advection:
            # Add V*div(w_h*grad(T)) implicitly: moves advection to LHS (non-symmetric).
            lhs_w = (
                lhs_w
                - dt
                * V_param
                * ufl.inner(w_trial * ufl.grad(T_ufl), ufl.grad(w_test))
                * ufl.dx
            )
        else:
            # Explicit advection using w_n: LHS stays SPD but requires CFL < 1.
            rhs_w = (
                rhs_w
                - dt
                * V_param
                * ufl.inner(self.w_n * ufl.grad(T_ufl), ufl.grad(w_test))
                * ufl.dx
            )

        # n-equation: (n_h - n_n)/dt = w_n*n_n^2 - m*n_h + Dn*laplacian(n_h)
        #
        # Implicit: diffusion Dn*laplacian(n_h), linear mortality -m*n_h
        # Explicit: reaction w_n*n_n^2
        #
        # LHS is constant and SPD: mass + dt*(mortality_mass + Dn*stiffness)
        lhs_n = (
            ufl.inner(n_trial, n_test) * ufl.dx
            + dt * m * ufl.inner(n_trial, n_test) * ufl.dx
            + dt * Dn * ufl.inner(ufl.grad(n_trial), ufl.grad(n_test)) * ufl.dx
        )
        rhs_n = (
            ufl.inner(self.n_n, n_test) * ufl.dx
            + dt * ufl.inner(self.w_n * self.n_n**2, n_test) * ufl.dx
        )

        self._dynamic_noise_runtimes = []
        w_stochastic_term, w_stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="w",
            previous_state=self.w_n,
            test=w_test,
            dt=dt,
        )
        if w_stochastic_term is not None and w_stochastic_runtime is not None:
            rhs_w = rhs_w + w_stochastic_term
            self._dynamic_noise_runtimes.append(w_stochastic_runtime)

        n_stochastic_term, n_stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="n",
            previous_state=self.n_n,
            test=n_test,
            dt=dt,
        )
        if n_stochastic_term is not None and n_stochastic_runtime is not None:
            rhs_n = rhs_n + n_stochastic_term
            self._dynamic_noise_runtimes.append(n_stochastic_runtime)

        lhs_w_bc, rhs_w_bc = build_natural_bc_forms(
            w_trial, w_test, domain_geom, w_boundary_field, self.config.parameters
        )
        if lhs_w_bc is not None:
            lhs_w = lhs_w + dt * lhs_w_bc
        if rhs_w_bc is not None:
            rhs_w = rhs_w + dt * rhs_w_bc

        lhs_n_bc, rhs_n_bc = build_natural_bc_forms(
            n_trial, n_test, domain_geom, n_boundary_field, self.config.parameters
        )
        if lhs_n_bc is not None:
            lhs_n = lhs_n + dt * lhs_n_bc
        if rhs_n_bc is not None:
            rhs_n = rhs_n + dt * rhs_n_bc

        self._w_problem = self.create_linear_problem(
            lhs_w,
            rhs_w,
            u=self.w_h,
            bcs=w_bcs,
            petsc_options_prefix="plm_klausmeier_topography_w_",
            mpc=w_mpc,
        )
        self._n_problem = self.create_linear_problem(
            lhs_n,
            rhs_n,
            u=self.n_h,
            bcs=n_bcs,
            petsc_options_prefix="plm_klausmeier_topography_n_",
            mpc=n_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._w_problem.solve()
        self._n_problem.solve()

        self.w_n.x.array[:] = self.w_h.x.array
        self.w_n.x.scatter_forward()
        self.n_n.x.array[:] = self.n_h.x.array
        self.n_n.x.scatter_forward()

        return (
            self._w_problem.solver.getConvergedReason() > 0
            and self._n_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"w": self.w_n, "n": self.n_n, "topography": self.topography_out}

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("klausmeier_topography")
class KlausmeierTopographyPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _KLAUSMEIER_TOPOGRAPHY_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _KlausmeierTopographyProblem(self.spec, config)
