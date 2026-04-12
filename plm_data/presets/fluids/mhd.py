"""Incompressible resistive magnetohydrodynamics preset."""

import ufl
from dolfinx import default_real_type, fem

from plm_data.core.boundary_conditions import build_vector_natural_bc_forms
from plm_data.core.fem_utils import domain_average
from plm_data.core.initial_conditions import apply_vector_ic
from plm_data.core.solver_strategies import TRANSIENT_MIXED_DIRECT
from plm_data.core.source_terms import build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_vector_standard_boundary_field,
)
from plm_data.presets.fluids._divfree_block import create_divergence_free_pair
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_MHD_SPEC = PresetSpec(
    name="mhd",
    category="fluids",
    description=(
        "Incompressible resistive magnetohydrodynamics with coupled velocity, "
        "pressure, magnetic field, and divergence-constraint dynamics."
    ),
    equations={
        "velocity": (
            "du/dt + (u_prev.grad)u - (B_prev.grad)B = -grad(p) "
            "+ (1/Re)*laplacian(u) + f_u"
        ),
        "pressure": "div(u) = 0",
        "magnetic_field": (
            "dB/dt + (u_prev.grad)B - (B_prev.grad)u = -grad(psi) "
            "+ (1/Rm)*laplacian(B) + f_B"
        ),
        "magnetic_constraint": "div(B) = 0",
    },
    parameters=[
        PDEParameter("Re", "Fluid Reynolds number"),
        PDEParameter("Rm", "Magnetic Reynolds number"),
        PDEParameter("k", "Constraint-space degree; vector spaces use k+1"),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "magnetic_field": InputSpec(
            name="magnetic_field",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the velocity field.",
        ),
        "magnetic_field": BoundaryFieldSpec(
            name="magnetic_field",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the magnetic field.",
        ),
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
        "magnetic_field": StateSpec(name="magnetic_field", shape="vector"),
        "magnetic_constraint": StateSpec(name="magnetic_constraint", shape="scalar"),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
        "magnetic_field": OutputSpec(
            name="magnetic_field",
            shape="vector",
            output_mode="components",
            source_name="magnetic_field",
        ),
        "magnetic_constraint": OutputSpec(
            name="magnetic_constraint",
            shape="scalar",
            output_mode="scalar",
            source_name="magnetic_constraint",
        ),
    },
    static_fields=[],
    supported_dimensions=[2, 3],
)


def _positive_parameter(parameters: dict[str, float], name: str) -> float:
    value = parameters[name]
    if value <= 0.0:
        raise ValueError(f"Preset 'mhd' requires parameter '{name}' > 0. Got {value}.")
    return value


def _scalar_degree(parameters: dict[str, float]) -> int:
    raw_degree = parameters["k"]
    degree = int(raw_degree)
    if float(degree) != float(raw_degree) or degree < 1:
        raise ValueError(
            f"Preset 'mhd' requires integer parameter 'k' >= 1. Got {raw_degree}."
        )
    return degree


def _normalize_scalar_lagrange_multiplier(msh, field: fem.Function) -> None:
    field.x.array[:] -= domain_average(msh, field)
    field.x.scatter_forward()


class _MHDProblem(TransientLinearProblem):
    supported_solver_strategies = (TRANSIENT_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        velocity_boundary_field = self.config.boundary_field("velocity")
        magnetic_boundary_field = self.config.boundary_field("magnetic_field")

        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="velocity",
            boundary_field=velocity_boundary_field,
            domain_geom=domain_geom,
        )
        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="magnetic_field",
            boundary_field=magnetic_boundary_field,
            domain_geom=domain_geom,
        )

        if (
            velocity_boundary_field.periodic_pair_keys()
            != magnetic_boundary_field.periodic_pair_keys()
        ):
            raise ValueError(
                "Velocity and magnetic-field boundary conditions must use identical "
                "periodic side pairs."
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.domain_geom = domain_geom
        self.msh = domain_geom.mesh

        parameters = self.config.parameters
        Re = _positive_parameter(parameters, "Re")
        Rm = _positive_parameter(parameters, "Rm")
        k = _scalar_degree(parameters)
        dt = self.config.time.dt

        velocity_input = self.config.input("velocity")
        magnetic_input = self.config.input("magnetic_field")
        velocity_boundary_field = self.config.boundary_field("velocity")
        magnetic_boundary_field = self.config.boundary_field("magnetic_field")

        flow_system = create_divergence_free_pair(
            self.create_periodic_constraint,
            domain_geom,
            velocity_boundary_field,
            parameters,
            scalar_degree=k,
        )
        magnetic_system = create_divergence_free_pair(
            self.create_periodic_constraint,
            domain_geom,
            magnetic_boundary_field,
            parameters,
            scalar_degree=k,
        )

        self._num_dofs = flow_system.num_dofs() + magnetic_system.num_dofs()

        self.u_h = flow_system.create_primary_function("velocity")
        self.p_h = flow_system.create_constraint_function("pressure")
        self.B_h = magnetic_system.create_primary_function("magnetic_field")
        self.psi_h = magnetic_system.create_constraint_function("magnetic_constraint")
        self.u_n = flow_system.create_primary_function("velocity_prev")
        self.B_n = magnetic_system.create_primary_function("magnetic_field_prev")

        assert velocity_input.initial_condition is not None
        apply_vector_ic(
            self.u_h,
            velocity_input.initial_condition,
            parameters,
            seed=self.config.seed,
        )
        self.u_h.x.scatter_forward()
        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()

        assert magnetic_input.initial_condition is not None
        apply_vector_ic(
            self.B_h,
            magnetic_input.initial_condition,
            parameters,
            seed=self.config.seed,
            stream_id="magnetic_field",
        )
        self.B_h.x.scatter_forward()
        self.B_n.x.array[:] = self.B_h.x.array
        self.B_n.x.scatter_forward()

        self.p_h.x.array[:] = 0.0
        self.p_h.x.scatter_forward()
        self.psi_h.x.array[:] = 0.0
        self.psi_h.x.scatter_forward()

        u = ufl.TrialFunction(flow_system.V)
        p = ufl.TrialFunction(flow_system.Q)
        B = ufl.TrialFunction(magnetic_system.V)
        psi = ufl.TrialFunction(magnetic_system.Q)
        v = ufl.TestFunction(flow_system.V)
        q = ufl.TestFunction(flow_system.Q)
        c = ufl.TestFunction(magnetic_system.V)
        r = ufl.TestFunction(magnetic_system.Q)

        delta_t = fem.Constant(self.msh, default_real_type(dt))
        zero_scalar = fem.Constant(self.msh, default_real_type(0.0))

        a_blocks = [
            [
                ufl.inner(u / delta_t, v) * ufl.dx
                + ufl.inner(ufl.dot(ufl.grad(u), self.u_n), v) * ufl.dx
                + default_real_type(1.0 / Re)
                * ufl.inner(ufl.grad(u), ufl.grad(v))
                * ufl.dx,
                -ufl.inner(p, ufl.div(v)) * ufl.dx,
                -ufl.inner(ufl.dot(ufl.grad(B), self.B_n), v) * ufl.dx,
                None,
            ],
            [
                -ufl.inner(ufl.div(u), q) * ufl.dx,
                None,
                None,
                None,
            ],
            [
                -ufl.inner(ufl.dot(ufl.grad(u), self.B_n), c) * ufl.dx,
                None,
                ufl.inner(B / delta_t, c) * ufl.dx
                + ufl.inner(ufl.dot(ufl.grad(B), self.u_n), c) * ufl.dx
                + default_real_type(1.0 / Rm)
                * ufl.inner(ufl.grad(B), ufl.grad(c))
                * ufl.dx,
                -ufl.inner(psi, ufl.div(c)) * ufl.dx,
            ],
            [
                None,
                None,
                -ufl.inner(ufl.div(B), r) * ufl.dx,
                None,
            ],
        ]

        L_blocks = [
            ufl.inner(self.u_n / delta_t, v) * ufl.dx,
            ufl.inner(zero_scalar, q) * ufl.dx,
            ufl.inner(self.B_n / delta_t, c) * ufl.dx,
            ufl.inner(zero_scalar, r) * ufl.dx,
        ]

        assert velocity_input.source is not None
        velocity_source_form = build_vector_source_form(
            v,
            self.msh,
            velocity_input.source,
            parameters,
        )
        if velocity_source_form is not None:
            L_blocks[0] = L_blocks[0] + velocity_source_form

        velocity_natural_form = build_vector_natural_bc_forms(
            v,
            domain_geom,
            velocity_boundary_field,
            parameters,
        )
        if velocity_natural_form is not None:
            L_blocks[0] = L_blocks[0] + velocity_natural_form

        assert magnetic_input.source is not None
        magnetic_source_form = build_vector_source_form(
            c,
            self.msh,
            magnetic_input.source,
            parameters,
        )
        if magnetic_source_form is not None:
            L_blocks[2] = L_blocks[2] + magnetic_source_form

        magnetic_natural_form = build_vector_natural_bc_forms(
            c,
            domain_geom,
            magnetic_boundary_field,
            parameters,
        )
        if magnetic_natural_form is not None:
            L_blocks[2] = L_blocks[2] + magnetic_natural_form

        bcs = flow_system.bcs + magnetic_system.bcs
        if flow_system.mpc_primary is None:
            mpc = None
        else:
            assert flow_system.mpc_constraint is not None
            assert magnetic_system.mpc_primary is not None
            assert magnetic_system.mpc_constraint is not None
            mpc = [
                flow_system.mpc_primary,
                flow_system.mpc_constraint,
                magnetic_system.mpc_primary,
                magnetic_system.mpc_constraint,
            ]

        self._problem = self.create_linear_problem(
            a_blocks,
            L_blocks,
            u=[self.u_h, self.p_h, self.B_h, self.psi_h],
            bcs=bcs,
            petsc_options_prefix="plm_mhd_",
            kind="mpi",
            mpc=mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._problem.solve()
        _normalize_scalar_lagrange_multiplier(self.msh, self.p_h)
        _normalize_scalar_lagrange_multiplier(self.msh, self.psi_h)

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        self.B_n.x.array[:] = self.B_h.x.array
        self.B_n.x.scatter_forward()

        return self._problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {
            "velocity": self.u_h,
            "pressure": self.p_h,
            "magnetic_field": self.B_h,
            "magnetic_constraint": self.psi_h,
        }

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("mhd")
class MHDPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _MHD_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _MHDProblem(self.spec, config)
