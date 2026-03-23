"""Cahn-Hilliard equation preset: phase separation via nonlinear 4th-order PDE.

Uses mixed finite elements (c, mu) with theta-method time stepping and
NonlinearProblem (Newton solver). Follows the DOLFINx Cahn-Hilliard demo.
"""

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from dolfinx.fem.petsc import NonlinearProblem

from plm_data.core.config import SimulationConfig
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_domain
from plm_data.presets import register_preset
from plm_data.presets.base import TimeDependentPreset
from plm_data.presets.metadata import PDEMetadata, PDEParameter


@register_preset("cahn_hilliard")
class CahnHilliardPreset(TimeDependentPreset):
    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cahn_hilliard",
            category="physics",
            description=(
                "Cahn-Hilliard equation for phase separation. "
                "Fourth-order nonlinear PDE split into two coupled second-order "
                "equations using a mixed formulation (concentration c and "
                "chemical potential mu)."
            ),
            equations={
                "c": "dc/dt = div(M * grad(mu))",
                "mu": "mu = df/dc - lambda * laplacian(c)",
            },
            parameters=[
                PDEParameter("lmbda", "Surface parameter (interface width)"),
                PDEParameter(
                    "theta",
                    "Time stepping parameter (0.5=Crank-Nicolson, 1=backward Euler)",
                ),
            ],
            field_names=["c"],
            steady_state=False,
            supported_dimensions=[2],
        )

    def setup(self, config: SimulationConfig) -> None:
        domain_geom = create_domain(config.domain)
        self.msh = domain_geom.mesh

        # Mixed function space: two P1 Lagrange elements for (c, mu)
        P1 = element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        ME = fem.functionspace(self.msh, mixed_element([P1, P1]))

        # Parameters
        lmbda = config.parameters["lmbda"]
        theta = config.parameters["theta"]
        dt = config.dt

        # Current and previous solution on mixed space
        self.u = fem.Function(ME)
        self.u0 = fem.Function(ME)

        # Apply initial condition to concentration component (sub 0)
        apply_ic(self.u.sub(0), config.initial_conditions["c"], seed=config.seed)
        self.u.x.scatter_forward()
        self.u0.x.array[:] = self.u.x.array

        # Split into components for variational form
        c, mu = ufl.split(self.u)
        c0, mu0 = ufl.split(self.u0)

        # Test functions
        q, v = ufl.TestFunctions(ME)

        # Chemical potential: f = 100*c^2*(1-c)^2, df/dc via automatic differentiation
        c = ufl.variable(c)
        f = 100 * c**2 * (1 - c) ** 2
        dfdc = ufl.diff(f, c)

        # Theta-method: mu at mid-point
        mu_mid = (1.0 - theta) * mu0 + theta * mu

        # Time step as constant
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        # Weak form (nonlinear residual)
        # Equation 1: dc/dt = div(M * grad(mu)), M=1
        F0 = (
            ufl.inner(c, q) * ufl.dx
            - ufl.inner(c0, q) * ufl.dx
            + dt_c * ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
        )
        # Equation 2: mu = df/dc - lmbda * laplacian(c)
        F1 = (
            ufl.inner(mu, v) * ufl.dx
            - ufl.inner(dfdc, v) * ufl.dx
            - lmbda * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
        )
        F = F0 + F1

        # Nonlinear problem with Newton solver
        self.problem = NonlinearProblem(
            F,
            self.u,
            petsc_options_prefix="plm_cahn_hilliard_",
            petsc_options=self._solver_options,
        )

        # Prepare output: collapse concentration sub-space
        V0, self._c_dofs = ME.sub(0).collapse()
        self.c_out = fem.Function(V0, name="c")

    def step(self, t: float, dt: float) -> None:
        # Copy current solution to previous
        self.u0.x.array[:] = self.u.x.array
        # Solve nonlinear system
        self.problem.solve()

    def get_output_fields(self) -> dict[str, fem.Function]:
        self.c_out.x.array[:] = self.u.x.array[self._c_dofs]
        return {"c": self.c_out}

    def get_num_dofs(self) -> int:
        return self.u.function_space.dofmap.index_map.size_global
