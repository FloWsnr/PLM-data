"""Swift-Hohenberg equation PDE."""

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem

from plm_data.boundary_conditions import get_boundary_operator_spec
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.runtime_config import BoundaryFieldConfig
from plm_data.core.solver_strategies import NONLINEAR_MIXED_DIRECT
from plm_data.core.spatial_fields import build_vector_ufl_field
from plm_data.pdes.base import PDE, ProblemInstance, TransientNonlinearProblem
from plm_data.pdes.boundary_validation import validate_boundary_field_structure
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_SWIFT_HOHENBERG_BOUNDARY_OPERATORS = {
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
    "simply_supported": get_boundary_operator_spec("simply_supported"),
}

_SWIFT_HOHENBERG_SPEC = PDESpec(
    name="swift_hohenberg",
    category="physics",
    description=(
        "Swift-Hohenberg equation for pattern formation using a mixed u/w "
        "formulation where w = (q0^2 + laplacian)(u)."
    ),
    equations={
        "u": (
            "du/dt + velocity.grad(u) = r*u - q0^2*w - laplacian(w) "
            "+ alpha*u^2 + beta*u^3 + gamma*u^5"
        ),
        "w": "w = q0^2*u + laplacian(u)",
    },
    parameters=[
        PDEParameter("r", "Bifurcation parameter"),
        PDEParameter("q0", "Critical wavenumber"),
        PDEParameter("alpha", "Quadratic nonlinearity coefficient"),
        PDEParameter("beta", "Cubic nonlinearity coefficient"),
        PDEParameter("gamma", "Quintic nonlinearity coefficient"),
        PDEParameter("theta", "Time-stepping parameter"),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        )
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=_SWIFT_HOHENBERG_BOUNDARY_OPERATORS,
            description="Boundary conditions for the primary field u.",
        )
    },
    states={
        "u": StateSpec(name="u", shape="scalar"),
        "w": StateSpec(name="w", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "velocity": CoefficientSpec(
            name="velocity",
            shape="vector",
            description="Prescribed advection velocity for the primary field u.",
        )
    },
)


def _boundary_mode(boundary_field: BoundaryFieldConfig) -> str:
    boundary_types = {
        entry.type for entries in boundary_field.sides.values() for entry in entries
    }
    if len(boundary_types) != 1:
        raise ValueError(
            "Swift-Hohenberg requires one global boundary mode: all sides must "
            "be periodic or all sides must be simply_supported."
        )
    return next(iter(boundary_types))


def _build_mixed_zero_dirichlet_bcs(
    mixed_space: fem.FunctionSpace,
    domain_geom,
    boundary_field: BoundaryFieldConfig,
) -> list[fem.DirichletBC]:
    bcs: list[fem.DirichletBC] = []
    fdim = domain_geom.mesh.topology.dim - 1

    for subspace in (mixed_space.sub(0), mixed_space.sub(1)):
        for side_name, entries in boundary_field.sides.items():
            if entries[0].type != "simply_supported":
                continue
            facets = domain_geom.facet_tags.find(domain_geom.boundary_names[side_name])
            dofs = fem.locate_dofs_topological(subspace, fdim, facets)
            bcs.append(fem.dirichletbc(default_real_type(0.0), dofs, subspace))

    return bcs


class _SwiftHohenbergProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        validate_boundary_field_structure(
            pde_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
            allowed_operators={"periodic", "simply_supported"},
        )
        _boundary_mode(self.config.boundary_field("u"))

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        boundary_field = self.config.boundary_field("u")

        P1 = element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        ME = fem.functionspace(self.msh, mixed_element([P1, P1]))

        assert self.config.time is not None
        r = self.config.parameters["r"]
        q0 = self.config.parameters["q0"]
        alpha = self.config.parameters["alpha"]
        beta = self.config.parameters["beta"]
        gamma = self.config.parameters["gamma"]
        theta = self.config.parameters["theta"]
        dt = self.config.time.dt
        if q0 <= 0.0:
            raise ValueError("Swift-Hohenberg q0 must be positive.")
        if not 0.0 <= theta <= 1.0:
            raise ValueError("Swift-Hohenberg theta must be between 0 and 1.")
        velocity = build_vector_ufl_field(
            self.msh,
            self.config.coefficient("velocity"),
            self.config.parameters,
        )
        if velocity is None:
            raise ValueError(
                "Swift-Hohenberg coefficient 'velocity' cannot use a custom expression"
            )

        boundary_mode = _boundary_mode(boundary_field)
        bcs: list[fem.DirichletBC] = []
        if boundary_mode == "periodic":
            mpc = self.create_periodic_constraint(
                ME,
                domain_geom,
                boundary_field,
                [],
                constrained_spaces=[ME.sub(0), ME.sub(1)],
            )
        else:
            mpc = None
            bcs = _build_mixed_zero_dirichlet_bcs(ME, domain_geom, boundary_field)

        solution_space = ME if mpc is None else mpc.function_space

        self.u = fem.Function(solution_space)
        self.u0 = fem.Function(solution_space)

        initial_condition = self.config.input("u").initial_condition
        assert initial_condition is not None
        apply_ic(
            self.u.sub(0),
            initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        for bc in bcs:
            bc.set(self.u.x.array)
        self.u.x.scatter_forward()
        self.u0.x.array[:] = self.u.x.array
        for bc in bcs:
            bc.set(self.u0.x.array)
        self.u0.x.scatter_forward()

        u, w = ufl.split(self.u)
        u0, w0 = ufl.split(self.u0)
        phi, psi = ufl.TestFunctions(ME)

        w_theta = (1.0 - theta) * w0 + theta * w
        dt_c = fem.Constant(self.msh, default_real_type(dt))
        N_u = alpha * u**2 + beta * u**3 + gamma * u**5

        F0 = (
            ufl.inner(u, phi) * ufl.dx
            - ufl.inner(u0, phi) * ufl.dx
            + dt_c * ufl.inner(velocity, ufl.grad(u)) * phi * ufl.dx
            - dt_c * r * ufl.inner(u, phi) * ufl.dx
            + dt_c * q0**2 * ufl.inner(w_theta, phi) * ufl.dx
            - dt_c * ufl.inner(ufl.grad(w_theta), ufl.grad(phi)) * ufl.dx
            - dt_c * ufl.inner(N_u, phi) * ufl.dx
        )
        F1 = (
            ufl.inner(w, psi) * ufl.dx
            - q0**2 * ufl.inner(u, psi) * ufl.dx
            + ufl.inner(ufl.grad(u), ufl.grad(psi)) * ufl.dx
        )
        F = F0 + F1
        J = ufl.derivative(F, self.u, ufl.TrialFunction(ME))

        self.problem = self.create_nonlinear_problem(
            F,
            self.u,
            bcs=bcs,
            petsc_options_prefix="plm_sh_",
            J=J,
            mpc=mpc,
        )

        V0, self._u_dofs = self.u.function_space.sub(0).collapse()
        self.u_out = fem.Function(V0, name="u")

    def step(self, t: float, dt: float) -> bool:
        self.u.x.scatter_forward()
        self.u0.x.array[:] = self.u.x.array
        self.u0.x.scatter_forward()
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason() > 0
        if converged:
            self.u.x.scatter_forward()
        return converged

    def get_output_fields(self) -> dict[str, fem.Function]:
        self.u.x.scatter_forward()
        self.u_out.x.array[:] = self.u.x.array[self._u_dofs]
        self.u_out.x.scatter_forward()
        return {"u": self.u_out}

    def get_num_dofs(self) -> int:
        return self.u.function_space.dofmap.index_map.size_global


class SwiftHohenbergPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _SWIFT_HOHENBERG_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _SwiftHohenbergProblem(self.spec, config)
