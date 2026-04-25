"""2D shallow-water equations with height anomaly and velocity."""

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from mpi4py import MPI

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs_to_subspace,
    apply_vector_dirichlet_bcs_to_subspace,
    build_vector_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic, apply_vector_ic
from plm_data.core.solver_strategies import NONLINEAR_MIXED_DIRECT
from plm_data.core.spatial_fields import (
    build_ufl_field,
    resolve_param_ref,
    scalar_expression_to_config,
)
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
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_SHALLOW_WATER_HEIGHT_BOUNDARY_OPERATORS = {
    "dirichlet": SCALAR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": SCALAR_STANDARD_BOUNDARY_OPERATORS["neumann"],
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
}

_SHALLOW_WATER_VELOCITY_BOUNDARY_OPERATORS = {
    "dirichlet": VECTOR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": VECTOR_STANDARD_BOUNDARY_OPERATORS["neumann"],
    "periodic": VECTOR_STANDARD_BOUNDARY_OPERATORS["periodic"],
}

_SHALLOW_WATER_SPEC = PDESpec(
    name="shallow_water",
    category="fluids",
    description="Two-dimensional shallow-water equations for height and velocity.",
    equations={
        "height": "d(height)/dt + div((mean_depth + height) * velocity) = 0",
        "velocity": (
            "d(velocity)/dt + (velocity.grad)velocity + gravity*grad(height + "
            "bathymetry) = -drag*velocity + viscosity*laplacian(velocity) "
            "- coriolis*perp(velocity)"
        ),
    },
    parameters=[
        PDEParameter("gravity", "Gravitational acceleration coefficient"),
        PDEParameter("mean_depth", "Background water depth"),
        PDEParameter("drag", "Linear bottom-friction coefficient"),
        PDEParameter("viscosity", "Velocity diffusion coefficient"),
        PDEParameter("coriolis", "Coriolis rotation coefficient"),
    ],
    inputs={
        "height": InputSpec(
            name="height",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "height": BoundaryFieldSpec(
            name="height",
            shape="scalar",
            operators=_SHALLOW_WATER_HEIGHT_BOUNDARY_OPERATORS,
            description="Boundary conditions for the height anomaly.",
        ),
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=_SHALLOW_WATER_VELOCITY_BOUNDARY_OPERATORS,
            description="Boundary conditions for depth-averaged velocity.",
        ),
    },
    states={
        "height": StateSpec(name="height", shape="scalar"),
        "velocity": StateSpec(name="velocity", shape="vector"),
    },
    outputs={
        "height": OutputSpec(
            name="height",
            shape="scalar",
            output_mode="scalar",
            source_name="height",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "bathymetry": CoefficientSpec(
            name="bathymetry",
            shape="scalar",
            description="Bed-elevation field added to the hydrostatic pressure term.",
        )
    },
)


def _perp(u):
    return ufl.as_vector((-u[1], u[0]))


def _is_zero_scalar_boundary_value(value_expr, parameters: dict[str, float]) -> bool:
    field_config = scalar_expression_to_config(value_expr)
    field_type = field_config["type"]
    if field_type in ("none", "zero"):
        return True
    if field_type != "constant":
        return False
    return (
        abs(resolve_param_ref(field_config["params"]["value"], parameters)) <= 1.0e-12
    )


class _ShallowWaterProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        if domain_geom.mesh.geometry.dim != 2:
            raise ValueError(
                f"PDE '{self.spec.name}' only supports 2D domains, got "
                f"{domain_geom.mesh.geometry.dim}D."
            )

        height_boundary_field = self.config.boundary_field("height")
        velocity_boundary_field = self.config.boundary_field("velocity")

        validate_boundary_field_structure(
            pde_name=self.spec.name,
            field_name="height",
            boundary_field=height_boundary_field,
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "neumann", "periodic"},
        )
        validate_boundary_field_structure(
            pde_name=self.spec.name,
            field_name="velocity",
            boundary_field=velocity_boundary_field,
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "neumann", "periodic"},
        )

        if (
            height_boundary_field.periodic_pair_keys()
            != velocity_boundary_field.periodic_pair_keys()
        ):
            raise ValueError(
                "Height and velocity boundary conditions must use identical "
                "periodic side pairs."
            )

        for entries in height_boundary_field.sides.values():
            for entry in entries:
                if entry.type != "neumann":
                    continue
                if entry.value is None or not _is_zero_scalar_boundary_value(
                    entry.value,
                    self.config.parameters,
                ):
                    raise ValueError(
                        "Shallow-water height Neumann boundaries currently only "
                        "support a zero-valued free boundary."
                    )

    def _update_output_views(self) -> None:
        self.height_out.x.array[:] = self.solution.x.array[self._height_dofs]
        self.height_out.x.scatter_forward()
        self.velocity_out.x.array[:] = self.solution.x.array[self._velocity_dofs]
        self.velocity_out.x.scatter_forward()

    def _positive_depth_metrics(self, context: str) -> dict[str, float]:
        self._update_output_views()
        local_height = self.height_out.x.array
        local_min_height = (
            float(np.min(local_height)) if local_height.size > 0 else float("inf")
        )
        min_depth = self.msh.comm.allreduce(
            self.mean_depth + local_min_height,
            op=MPI.MIN,
        )
        if min_depth <= 0.0:
            raise ValueError(
                f"PDE '{self.spec.name}' produced non-positive total depth "
                f"during {context}. Minimum depth: {min_depth:.6g}."
            )
        return {"min_total_depth": min_depth}

    def runtime_health_metrics(self, context: str) -> dict[str, float]:
        return self._positive_depth_metrics(context)

    def setup(self) -> None:
        assert self.config.time is not None
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        self.mean_depth = self.config.parameters["mean_depth"]
        if self.mean_depth <= 0.0:
            raise ValueError("Shallow-water mean_depth must be positive.")

        height_boundary_field = self.config.boundary_field("height")
        velocity_boundary_field = self.config.boundary_field("velocity")
        scalar_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            1,
            dtype=default_real_type,
        )
        vector_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            1,
            shape=(self.msh.geometry.dim,),
            dtype=default_real_type,
        )
        mixed_space = fem.functionspace(
            self.msh,
            mixed_element([scalar_element, vector_element]),
        )

        bcs = [
            *apply_dirichlet_bcs_to_subspace(
                mixed_space.sub(0),
                domain_geom,
                height_boundary_field,
                self.config.parameters,
            ),
            *apply_vector_dirichlet_bcs_to_subspace(
                mixed_space.sub(1),
                domain_geom,
                velocity_boundary_field,
                self.config.parameters,
            ),
        ]
        mpc = self.create_periodic_constraint(
            mixed_space,
            domain_geom,
            height_boundary_field,
            bcs,
            constrained_spaces=[mixed_space.sub(0), mixed_space.sub(1)],
        )
        solution_space = mixed_space if mpc is None else mpc.function_space

        self.solution = fem.Function(solution_space)
        self.previous = fem.Function(solution_space)
        height_space, self._height_dofs = self.solution.function_space.sub(0).collapse()
        velocity_space, self._velocity_dofs = self.solution.function_space.sub(
            1
        ).collapse()
        self.height_out = fem.Function(height_space, name="height")
        self.velocity_out = fem.Function(velocity_space, name="velocity")

        height_input = self.config.input("height")
        velocity_input = self.config.input("velocity")
        assert height_input.initial_condition is not None
        assert velocity_input.initial_condition is not None
        apply_ic(
            self.solution.sub(0),
            height_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        apply_vector_ic(
            self.solution.sub(1),
            velocity_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.solution.x.scatter_forward()
        self.previous.x.array[:] = self.solution.x.array
        self.previous.x.scatter_forward()

        bathymetry = build_ufl_field(
            self.msh,
            scalar_expression_to_config(self.config.coefficient("bathymetry")),
            self.config.parameters,
        )
        if bathymetry is None:
            raise ValueError("Shallow-water coefficient 'bathymetry' cannot be custom")

        height, velocity = ufl.split(self.solution)
        height_prev, velocity_prev = ufl.split(self.previous)
        test_height, test_velocity = ufl.TestFunctions(mixed_space)
        trial = ufl.TrialFunction(mixed_space)

        gravity = fem.Constant(
            self.msh,
            default_real_type(self.config.parameters["gravity"]),
        )
        drag = fem.Constant(
            self.msh,
            default_real_type(self.config.parameters["drag"]),
        )
        viscosity = fem.Constant(
            self.msh,
            default_real_type(self.config.parameters["viscosity"]),
        )
        coriolis = fem.Constant(
            self.msh,
            default_real_type(self.config.parameters["coriolis"]),
        )
        delta_t = fem.Constant(self.msh, default_real_type(self.config.time.dt))
        total_depth = self.mean_depth + height

        F_height = (
            (height - height_prev) / delta_t
        ) * test_height * ufl.dx + test_height * ufl.div(
            total_depth * velocity
        ) * ufl.dx
        F_velocity = (
            ufl.inner((velocity - velocity_prev) / delta_t, test_velocity) * ufl.dx
            + ufl.inner(ufl.dot(ufl.grad(velocity), velocity), test_velocity) * ufl.dx
            + gravity * ufl.inner(ufl.grad(height + bathymetry), test_velocity) * ufl.dx
            + drag * ufl.inner(velocity, test_velocity) * ufl.dx
            + viscosity
            * ufl.inner(ufl.grad(velocity), ufl.grad(test_velocity))
            * ufl.dx
            + coriolis * ufl.inner(_perp(velocity), test_velocity) * ufl.dx
        )
        traction_form = build_vector_natural_bc_forms(
            test_velocity,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
        )
        if traction_form is not None:
            F_velocity = F_velocity - traction_form
        F = F_height + F_velocity
        J = ufl.derivative(F, self.solution, trial)

        self.problem = self.create_nonlinear_problem(
            F,
            self.solution,
            bcs=bcs,
            petsc_options_prefix="plm_shallow_water_",
            J=J,
            mpc=mpc,
        )
        self._update_output_views()

    def step(self, t: float, dt: float) -> bool:
        self.previous.x.array[:] = self.solution.x.array
        self.previous.x.scatter_forward()
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason() > 0
        if converged:
            self.solution.x.scatter_forward()
        return converged

    def get_output_fields(self) -> dict[str, fem.Function]:
        self._update_output_views()
        return {
            "height": self.height_out,
            "velocity": self.velocity_out,
        }

    def get_num_dofs(self) -> int:
        return self.solution.function_space.dofmap.index_map.size_global


class ShallowWaterPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _SHALLOW_WATER_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _ShallowWaterProblem(self.spec, config)
