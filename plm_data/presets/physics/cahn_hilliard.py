"""Cahn-Hilliard equation preset."""

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from dolfinx.fem.petsc import NonlinearProblem

from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_domain
from plm_data.presets import register_preset
from plm_data.presets.base import (
    PDEPreset,
    ProblemInstance,
    TransientNonlinearProblem,
)
from plm_data.presets.metadata import (
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
)

_CAHN_HILLIARD_SPEC = PresetSpec(
    name="cahn_hilliard",
    category="physics",
    description=(
        "Cahn-Hilliard equation for phase separation using a mixed "
        "concentration/chemical-potential formulation."
    ),
    equations={
        "c": "dc/dt = div(M * grad(mu))",
        "mu": "mu = df/dc - lambda * laplacian(c)",
    },
    parameters=[
        PDEParameter("lmbda", "Surface parameter (interface width)"),
        PDEParameter("barrier_height", "Height of the double-well free-energy barrier"),
        PDEParameter("mobility", "Mobility coefficient M"),
        PDEParameter("theta", "Time-stepping parameter"),
    ],
    inputs={
        "c": InputSpec(
            name="c",
            shape="scalar",
            allow_boundary_conditions=False,
            allow_source=False,
            allow_initial_condition=True,
        )
    },
    states={
        "c": StateSpec(name="c", shape="scalar"),
        "mu": StateSpec(name="mu", shape="scalar"),
    },
    outputs={
        "c": OutputSpec(
            name="c",
            shape="scalar",
            output_mode="scalar",
            source_name="c",
        )
    },
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _CahnHilliardProblem(TransientNonlinearProblem):
    def setup(self) -> None:
        domain_geom = create_domain(self.config.domain)
        self.msh = domain_geom.mesh

        P1 = element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        ME = fem.functionspace(self.msh, mixed_element([P1, P1]))

        lmbda = self.config.parameters["lmbda"]
        barrier_height = self.config.parameters["barrier_height"]
        mobility = self.config.parameters["mobility"]
        theta = self.config.parameters["theta"]
        dt = self.config.time.dt

        self.u = fem.Function(ME)
        self.u0 = fem.Function(ME)

        initial_condition = self.config.input("c").initial_condition
        assert initial_condition is not None
        apply_ic(
            self.u.sub(0),
            initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.u.x.scatter_forward()
        self.u0.x.array[:] = self.u.x.array

        c, mu = ufl.split(self.u)
        c0, mu0 = ufl.split(self.u0)
        q, v = ufl.TestFunctions(ME)

        c = ufl.variable(c)
        free_energy = barrier_height * c**2 * (1 - c) ** 2
        dfdc = ufl.diff(free_energy, c)
        mu_mid = (1.0 - theta) * mu0 + theta * mu
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        F0 = (
            ufl.inner(c, q) * ufl.dx
            - ufl.inner(c0, q) * ufl.dx
            + dt_c * mobility * ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
        )
        F1 = (
            ufl.inner(mu, v) * ufl.dx
            - ufl.inner(dfdc, v) * ufl.dx
            - lmbda * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
        )
        F = F0 + F1

        self.problem = NonlinearProblem(
            F,
            self.u,
            petsc_options_prefix="plm_cahn_hilliard_",
            petsc_options=self._solver_options,
        )

        V0, self._c_dofs = ME.sub(0).collapse()
        self.c_out = fem.Function(V0, name="c")

    def step(self, t: float, dt: float) -> bool:
        self.u0.x.array[:] = self.u.x.array
        self.problem.solve()
        return self.problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        self.c_out.x.array[:] = self.u.x.array[self._c_dofs]
        return {"c": self.c_out}

    def get_num_dofs(self) -> int:
        return self.u.function_space.dofmap.index_map.size_global


@register_preset("cahn_hilliard")
class CahnHilliardPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _CAHN_HILLIARD_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _CahnHilliardProblem(self.spec, config)
