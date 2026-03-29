"""Kirchhoff plate equation preset using a mixed second-order formulation."""

import numpy as np
import ufl
from dolfinx import default_real_type, fem

from plm_data.core.config import BoundaryFieldConfig
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import DomainGeometry
from plm_data.core.source_terms import build_source_form
from plm_data.core.spatial_fields import (
    build_ufl_field,
    resolve_param_ref,
    scalar_expression_to_config,
)
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import validate_boundary_field_structure
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    BoundaryOperatorSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
)

_PLATE_BOUNDARY_OPERATORS = {
    "simply_supported": BoundaryOperatorSpec(
        name="simply_supported",
        value_shape=None,
        description="Zero deflection and zero bending moment boundary condition.",
    ),
}

_PLATE_SPEC = PresetSpec(
    name="plate",
    category="basic",
    description=(
        "Kirchhoff plate equation using a mixed deflection/velocity/moment "
        "formulation with simply supported boundaries."
    ),
    equations={
        "deflection": "dw/dt = v",
        "velocity": "rho_h * dv/dt + damping * v - div(rigidity * grad(m)) = q",
        "moment": "m = -laplacian(w)",
    },
    parameters=[
        PDEParameter("theta", "Implicit time-stepping parameter in [0.5, 1.0]"),
    ],
    inputs={
        "deflection": InputSpec(
            name="deflection",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "load": InputSpec(
            name="load",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "deflection": BoundaryFieldSpec(
            name="deflection",
            shape="scalar",
            operators=_PLATE_BOUNDARY_OPERATORS,
            description="Plate edge conditions for the deflection field.",
        )
    },
    states={
        "deflection": StateSpec(name="deflection", shape="scalar"),
        "velocity": StateSpec(name="velocity", shape="scalar"),
        "moment": StateSpec(name="moment", shape="scalar"),
    },
    outputs={
        "deflection": OutputSpec(
            name="deflection",
            shape="scalar",
            output_mode="scalar",
            source_name="deflection",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="scalar",
            output_mode="scalar",
            source_name="velocity",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2],
    coefficients={
        "rho_h": CoefficientSpec(
            name="rho_h",
            shape="scalar",
            description="Mass per unit area.",
        ),
        "damping": CoefficientSpec(
            name="damping",
            shape="scalar",
            description="Viscous damping coefficient.",
        ),
        "rigidity": CoefficientSpec(
            name="rigidity",
            shape="scalar",
            description="Flexural rigidity coefficient.",
        ),
    },
)


def _locate_boundary_dofs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    name: str,
):
    """Locate scalar DOFs on one named boundary."""
    tag = domain_geom.boundary_names[name]
    facets = domain_geom.facet_tags.find(tag)
    return fem.locate_dofs_topological(
        V=V,
        entity_dim=domain_geom.mesh.topology.dim - 1,
        entities=facets,
    )


def _build_zero_dirichlet_bcs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    boundary_field: BoundaryFieldConfig,
) -> tuple[list[fem.DirichletBC], np.ndarray]:
    """Create zero Dirichlet BCs for every simply supported edge."""
    zero = fem.Constant(domain_geom.mesh, default_real_type(0.0))
    bcs: list[fem.DirichletBC] = []
    constrained_dofs: list[np.ndarray] = []

    for name, entries in boundary_field.sides.items():
        if name not in domain_geom.boundary_names:
            raise ValueError(
                f"Boundary '{name}' not found in domain. "
                f"Available boundaries: {list(domain_geom.boundary_names.keys())}"
            )
        if len(entries) != 1 or entries[0].type != "simply_supported":
            raise ValueError(
                f"Boundary '{name}' must contain exactly one "
                "'simply_supported' operator."
            )

        dofs = _locate_boundary_dofs(V, domain_geom, name)
        constrained_dofs.append(dofs)
        bcs.append(fem.dirichletbc(zero, dofs, V))

    if not constrained_dofs:
        return bcs, np.empty(0, dtype=np.int32)
    return bcs, np.unique(np.concatenate(constrained_dofs))


def _zero_constrained_dofs(func: fem.Function, dofs: np.ndarray) -> None:
    """Enforce homogeneous Dirichlet data on the selected DOFs."""
    if dofs.size == 0:
        return
    func.x.array[dofs] = 0.0
    func.x.scatter_forward()


def _build_scalar_coefficient(problem: ProblemInstance, name: str):
    """Build a scalar UFL coefficient declared in the preset spec."""
    coefficient_config = scalar_expression_to_config(problem.config.coefficient(name))
    coefficient_type = coefficient_config["type"]
    if coefficient_type in {"none", "zero"}:
        return fem.Constant(problem.msh, default_real_type(0.0))
    if coefficient_type == "constant":
        value = resolve_param_ref(
            coefficient_config["params"]["value"],
            problem.config.parameters,
        )
        return fem.Constant(problem.msh, default_real_type(value))

    coefficient = build_ufl_field(
        problem.msh,
        coefficient_config,
        problem.config.parameters,
    )
    if coefficient is None:
        raise ValueError(f"Plate coefficient '{name}' cannot use a custom expression")
    return coefficient


class _PlateProblem(TransientLinearProblem):
    def validate_boundary_conditions(self, domain_geom):
        if domain_geom.mesh.geometry.dim != 2:
            raise ValueError(
                f"Preset '{self.spec.name}' only supports 2D domains, got "
                f"{domain_geom.mesh.geometry.dim}D."
            )
        validate_boundary_field_structure(
            preset_name=self.spec.name,
            field_name="deflection",
            boundary_field=self.config.boundary_field("deflection"),
            domain_geom=domain_geom,
            allowed_operators={"simply_supported"},
        )

    def _solve_initial_moment(self, m_bcs: list[fem.DirichletBC]) -> None:
        """Initialize the mixed auxiliary field from the initial deflection."""
        m_trial = ufl.TrialFunction(self.V_m)
        eta = ufl.TestFunction(self.V_m)
        a_m = ufl.inner(m_trial, eta) * ufl.dx
        L_m = ufl.inner(ufl.grad(self.w_n), ufl.grad(eta)) * ufl.dx

        initializer = self.create_linear_problem(
            a_m,
            L_m,
            u=self.m_n,
            bcs=m_bcs,
            petsc_options_prefix="plm_plate_init_moment_",
        )
        initializer.solve()
        reason = initializer.solver.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                f"Plate initial moment solve did not converge (KSP reason={reason})"
            )

    def setup(self) -> None:
        theta = self.config.parameters["theta"]
        if not 0.5 <= theta <= 1.0:
            raise ValueError(
                f"Plate parameter 'theta' must lie in [0.5, 1.0]. Got {theta}."
            )

        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        self.V_w = fem.functionspace(self.msh, ("Lagrange", 1))
        self.V_v = fem.functionspace(self.msh, ("Lagrange", 1))
        self.V_m = fem.functionspace(self.msh, ("Lagrange", 1))

        boundary_field = self.config.boundary_field("deflection")
        w_bcs, w_dofs = _build_zero_dirichlet_bcs(self.V_w, domain_geom, boundary_field)
        v_bcs, v_dofs = _build_zero_dirichlet_bcs(self.V_v, domain_geom, boundary_field)
        m_bcs, _ = _build_zero_dirichlet_bcs(self.V_m, domain_geom, boundary_field)

        self.w_n = fem.Function(self.V_w, name="deflection")
        self.v_n = fem.Function(self.V_v, name="velocity")
        self.m_n = fem.Function(self.V_m, name="moment")

        self.w_h = fem.Function(self.V_w, name="deflection_next")
        self.v_h = fem.Function(self.V_v, name="velocity_next")
        self.m_h = fem.Function(self.V_m, name="moment_next")

        deflection_ic = self.config.input("deflection").initial_condition
        velocity_ic = self.config.input("velocity").initial_condition
        assert deflection_ic is not None
        assert velocity_ic is not None
        apply_ic(
            self.w_n,
            deflection_ic,
            self.config.parameters,
            seed=self.config.seed,
        )
        apply_ic(
            self.v_n,
            velocity_ic,
            self.config.parameters,
            seed=self.config.seed,
        )

        # Simply supported deflection is fixed in time, so velocity is zero there too.
        _zero_constrained_dofs(self.w_n, w_dofs)
        _zero_constrained_dofs(self.v_n, v_dofs)
        self._solve_initial_moment(m_bcs)

        self._num_dofs = sum(
            space.dofmap.index_map.size_global * space.dofmap.index_map_bs
            for space in (self.V_w, self.V_v, self.V_m)
        )

        rho_h = _build_scalar_coefficient(self, "rho_h")
        damping = _build_scalar_coefficient(self, "damping")
        rigidity = _build_scalar_coefficient(self, "rigidity")

        dt = self.config.time.dt
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        w = ufl.TrialFunction(self.V_w)
        phi = ufl.TestFunction(self.V_w)
        v = ufl.TrialFunction(self.V_v)
        psi = ufl.TestFunction(self.V_v)
        m = ufl.TrialFunction(self.V_m)
        eta = ufl.TestFunction(self.V_m)

        a_blocks = [
            [
                ufl.inner(w, phi) * ufl.dx,
                -theta * dt_c * ufl.inner(v, phi) * ufl.dx,
                None,
            ],
            [
                None,
                ufl.inner(rho_h * v, psi) * ufl.dx
                + theta * dt_c * ufl.inner(damping * v, psi) * ufl.dx,
                theta
                * dt_c
                * ufl.inner(rigidity * ufl.grad(m), ufl.grad(psi))
                * ufl.dx,
            ],
            [
                -ufl.inner(ufl.grad(w), ufl.grad(eta)) * ufl.dx,
                None,
                ufl.inner(m, eta) * ufl.dx,
            ],
        ]

        L_blocks = [
            ufl.inner(self.w_n, phi) * ufl.dx
            + (1.0 - theta) * dt_c * ufl.inner(self.v_n, phi) * ufl.dx,
            ufl.inner(rho_h * self.v_n, psi) * ufl.dx
            - (1.0 - theta) * dt_c * ufl.inner(damping * self.v_n, psi) * ufl.dx
            - (1.0 - theta)
            * dt_c
            * ufl.inner(rigidity * ufl.grad(self.m_n), ufl.grad(psi))
            * ufl.dx,
            ufl.inner(fem.Constant(self.msh, default_real_type(0.0)), eta) * ufl.dx,
        ]

        load_input = self.config.input("load")
        assert load_input.source is not None
        load_form = build_source_form(
            psi,
            self.msh,
            load_input.source,
            self.config.parameters,
        )
        if load_form is not None:
            L_blocks[1] = L_blocks[1] + dt_c * load_form

        self._problem = self.create_linear_problem(
            a_blocks,
            L_blocks,
            u=[self.w_h, self.v_h, self.m_h],
            bcs=w_bcs + v_bcs + m_bcs,
            petsc_options_prefix="plm_plate_",
            kind="mpi",
        )

    def step(self, t: float, dt: float) -> bool:
        self._problem.solve()

        self.w_n.x.array[:] = self.w_h.x.array
        self.w_n.x.scatter_forward()
        self.v_n.x.array[:] = self.v_h.x.array
        self.v_n.x.scatter_forward()
        self.m_n.x.array[:] = self.m_h.x.array
        self.m_n.x.scatter_forward()

        return self._problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {
            "deflection": self.w_n,
            "velocity": self.v_n,
        }

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("plate")
class PlatePreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _PLATE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _PlateProblem(self.spec, config)
