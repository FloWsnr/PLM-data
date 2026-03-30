"""Diffusively coupled Lorenz system reaction-diffusion preset."""

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
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

_LORENZ_SPEC = PresetSpec(
    name="lorenz",
    category="physics",
    description=(
        "Diffusively coupled Lorenz system. Three-species reaction-diffusion "
        "with chaotic local dynamics from coupled Lorenz oscillators."
    ),
    equations={
        "x": "dx/dt = D * laplacian(x) + sigma * (y - x)",
        "y": "dy/dt = D * laplacian(y) + x * (rho - z) - y",
        "z": "dz/dt = D * laplacian(z) + x * y - beta * z",
    },
    parameters=[
        PDEParameter("sigma", "Prandtl number"),
        PDEParameter("rho", "Normalized Rayleigh number"),
        PDEParameter("beta", "Geometric factor"),
        PDEParameter("D", "Diffusion (spatial coupling) coefficient"),
    ],
    inputs={
        "x": InputSpec(
            name="x",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "y": InputSpec(
            name="y",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "z": InputSpec(
            name="z",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "x": BoundaryFieldSpec(
            name="x",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for Lorenz x field.",
        ),
        "y": BoundaryFieldSpec(
            name="y",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for Lorenz y field.",
        ),
        "z": BoundaryFieldSpec(
            name="z",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for Lorenz z field.",
        ),
    },
    states={
        "x": StateSpec(name="x", shape="scalar"),
        "y": StateSpec(name="y", shape="scalar"),
        "z": StateSpec(name="z", shape="scalar"),
    },
    outputs={
        "x": OutputSpec(
            name="x",
            shape="scalar",
            output_mode="scalar",
            source_name="x",
        ),
        "y": OutputSpec(
            name="y",
            shape="scalar",
            output_mode="scalar",
            source_name="y",
        ),
        "z": OutputSpec(
            name="z",
            shape="scalar",
            output_mode="scalar",
            source_name="z",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _LorenzProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        for field_name in ("x", "y", "z"):
            validate_scalar_standard_boundary_field(
                preset_name=self.spec.name,
                field_name=field_name,
                boundary_field=self.config.boundary_field(field_name),
                domain_geom=domain_geom,
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        x_input = self.config.input("x")
        y_input = self.config.input("y")
        z_input = self.config.input("z")
        x_boundary_field = self.config.boundary_field("x")
        y_boundary_field = self.config.boundary_field("y")
        z_boundary_field = self.config.boundary_field("z")

        dt = self.config.time.dt
        sigma = self.config.parameters["sigma"]
        rho = self.config.parameters["rho"]
        beta = self.config.parameters["beta"]
        D = self.config.parameters["D"]

        # Dirichlet BCs
        x_bcs = apply_dirichlet_bcs(
            V, domain_geom, x_boundary_field, self.config.parameters
        )
        y_bcs = apply_dirichlet_bcs(
            V, domain_geom, y_boundary_field, self.config.parameters
        )
        z_bcs = apply_dirichlet_bcs(
            V, domain_geom, z_boundary_field, self.config.parameters
        )

        # Periodic constraints
        x_mpc = self.create_periodic_constraint(V, domain_geom, x_boundary_field, x_bcs)
        y_mpc = self.create_periodic_constraint(V, domain_geom, y_boundary_field, y_bcs)
        z_mpc = self.create_periodic_constraint(V, domain_geom, z_boundary_field, z_bcs)

        self.V_x = V if x_mpc is None else x_mpc.function_space
        self.V_y = V if y_mpc is None else y_mpc.function_space
        self.V_z = V if z_mpc is None else z_mpc.function_space
        self._num_dofs = (
            _space_num_dofs(self.V_x)
            + _space_num_dofs(self.V_y)
            + _space_num_dofs(self.V_z)
        )

        # Solution functions: current and next timestep
        self.x_n = fem.Function(self.V_x, name="x")
        self.y_n = fem.Function(self.V_y, name="y")
        self.z_n = fem.Function(self.V_z, name="z")
        self.x_h = fem.Function(self.V_x, name="x_next")
        self.y_h = fem.Function(self.V_y, name="y_next")
        self.z_h = fem.Function(self.V_z, name="z_next")

        # Initial conditions with seed offsets
        assert x_input.initial_condition is not None
        apply_ic(
            self.x_n,
            x_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        assert y_input.initial_condition is not None
        apply_ic(
            self.y_n,
            y_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
        )
        assert z_input.initial_condition is not None
        apply_ic(
            self.z_n,
            z_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 2) if self.config.seed is not None else None,
        )
        self.x_n.x.scatter_forward()
        self.y_n.x.scatter_forward()
        self.z_n.x.scatter_forward()

        # Trial and test functions
        x_trial = ufl.TrialFunction(V)
        x_test = ufl.TestFunction(V)
        y_trial = ufl.TrialFunction(V)
        y_test = ufl.TestFunction(V)
        z_trial = ufl.TrialFunction(V)
        z_test = ufl.TestFunction(V)

        # X equation: (X^{n+1} - X^n)/dt = D*nabla^2(X^{n+1}) + sigma*(Y^n - X^{n+1})
        a_x = (
            ufl.inner(x_trial, x_test) * ufl.dx
            + dt * D * ufl.inner(ufl.grad(x_trial), ufl.grad(x_test)) * ufl.dx
            + dt * sigma * ufl.inner(x_trial, x_test) * ufl.dx
        )
        L_x = (
            ufl.inner(self.x_n, x_test) * ufl.dx
            + dt * sigma * ufl.inner(self.y_n, x_test) * ufl.dx
        )

        # Y equation: (Y^{n+1} - Y^n)/dt = D*nabla^2(Y^{n+1}) + X^n*(rho - Z^n) - Y^{n+1}
        a_y = (
            ufl.inner(y_trial, y_test) * ufl.dx
            + dt * D * ufl.inner(ufl.grad(y_trial), ufl.grad(y_test)) * ufl.dx
            + dt * ufl.inner(y_trial, y_test) * ufl.dx
        )
        L_y = (
            ufl.inner(self.y_n, y_test) * ufl.dx
            + dt * ufl.inner(self.x_n * (rho - self.z_n), y_test) * ufl.dx
        )

        # Z equation: (Z^{n+1} - Z^n)/dt = D*nabla^2(Z^{n+1}) + X^n*Y^n - beta*Z^{n+1}
        a_z = (
            ufl.inner(z_trial, z_test) * ufl.dx
            + dt * D * ufl.inner(ufl.grad(z_trial), ufl.grad(z_test)) * ufl.dx
            + dt * beta * ufl.inner(z_trial, z_test) * ufl.dx
        )
        L_z = (
            ufl.inner(self.z_n, z_test) * ufl.dx
            + dt * ufl.inner(self.x_n * self.y_n, z_test) * ufl.dx
        )

        # Natural BC forms (Neumann/Robin)
        a_x_bc, L_x_bc = build_natural_bc_forms(
            x_trial, x_test, domain_geom, x_boundary_field, self.config.parameters
        )
        if a_x_bc is not None:
            a_x = a_x + dt * a_x_bc
        if L_x_bc is not None:
            L_x = L_x + dt * L_x_bc

        a_y_bc, L_y_bc = build_natural_bc_forms(
            y_trial, y_test, domain_geom, y_boundary_field, self.config.parameters
        )
        if a_y_bc is not None:
            a_y = a_y + dt * a_y_bc
        if L_y_bc is not None:
            L_y = L_y + dt * L_y_bc

        a_z_bc, L_z_bc = build_natural_bc_forms(
            z_trial, z_test, domain_geom, z_boundary_field, self.config.parameters
        )
        if a_z_bc is not None:
            a_z = a_z + dt * a_z_bc
        if L_z_bc is not None:
            L_z = L_z + dt * L_z_bc

        # Create linear solvers
        self._x_problem = self.create_linear_problem(
            a_x,
            L_x,
            u=self.x_h,
            bcs=x_bcs,
            petsc_options_prefix="plm_lorenz_x_",
            mpc=x_mpc,
        )
        self._y_problem = self.create_linear_problem(
            a_y,
            L_y,
            u=self.y_h,
            bcs=y_bcs,
            petsc_options_prefix="plm_lorenz_y_",
            mpc=y_mpc,
        )
        self._z_problem = self.create_linear_problem(
            a_z,
            L_z,
            u=self.z_h,
            bcs=z_bcs,
            petsc_options_prefix="plm_lorenz_z_",
            mpc=z_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._x_problem.solve()
        self._y_problem.solve()
        self._z_problem.solve()

        self.x_n.x.array[:] = self.x_h.x.array
        self.x_n.x.scatter_forward()
        self.y_n.x.array[:] = self.y_h.x.array
        self.y_n.x.scatter_forward()
        self.z_n.x.array[:] = self.z_h.x.array
        self.z_n.x.scatter_forward()

        return (
            self._x_problem.solver.getConvergedReason() > 0
            and self._y_problem.solver.getConvergedReason() > 0
            and self._z_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"x": self.x_n, "y": self.y_n, "z": self.z_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("lorenz")
class LorenzPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _LORENZ_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _LorenzProblem(self.spec, config)
