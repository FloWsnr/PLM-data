"""Heat equation preset: du/dt = kappa * laplacian(u)."""

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.config import SimulationConfig
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.mesh import create_mesh
from plm_data.presets import register_preset
from plm_data.presets.base import RunResult, TimeDependentPreset
from plm_data.presets.metadata import PDEMetadata, PDEParameter


@register_preset("heat")
class HeatPreset(TimeDependentPreset):

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heat",
            category="basic",
            description="Heat equation du/dt = kappa * laplacian(u)",
            equations={"u": "∂u/∂t = κ ∇²u"},
            parameters=[
                PDEParameter("kappa", "Thermal diffusivity"),
            ],
            field_names=["u"],
            steady_state=False,
            supported_dimensions=[2],
            recommended_config={
                "preset": "heat",
                "parameters": {"kappa": 0.01},
                "solver": {"ksp_type": "preonly", "pc_type": "lu"},
                "domain": {
                    "type": "rectangle",
                    "size": [1.0, 1.0],
                    "mesh_resolution": [128, 128],
                },
                "output_resolution": [64, 64],
                "initial_condition": {
                    "type": "gaussian_bump",
                    "params": {
                        "sigma": 0.1,
                        "amplitude": 1.0,
                        "cx": 0.5,
                        "cy": 0.5,
                    },
                },
                "dt": 0.01,
                "t_end": 1.0,
                "output": {
                    "path": "./output",
                    "num_frames": 20,
                    "formats": ["numpy"],
                },
                "seed": 42,
            },
        )

    def setup(self, config: SimulationConfig) -> None:
        self.msh = create_mesh(config.domain)
        self.V = fem.functionspace(self.msh, ("Lagrange", 1))

        kappa = config.parameters["kappa"]
        dt = config.dt

        # Current and previous solution
        self.u_n = fem.Function(self.V, name="u_n")
        self.uh = fem.Function(self.V, name="u")

        # Apply initial condition from config
        apply_ic(self.u_n, config.initial_condition, seed=config.seed)

        # Implicit Euler: (u - u_n)/dt = kappa * laplacian(u)
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        dt_c = fem.Constant(self.msh, np.float64(dt))
        kappa_c = fem.Constant(self.msh, np.float64(kappa))

        a = (
            ufl.inner(u, v) * ufl.dx
            + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        L = ufl.inner(self.u_n, v) * ufl.dx

        # No BCs = natural Neumann (zero-flux) on all boundaries
        self.problem = LinearProblem(
            a, L, bcs=[],
            petsc_options_prefix="plm_heat_",
            petsc_options=self._solver_options,
        )

    def step(self, t: float, dt: float) -> None:
        self.uh = self.problem.solve()
        self.u_n.x.array[:] = self.uh.x.array

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n}

    def get_num_dofs(self) -> int:
        return self.V.dofmap.index_map.size_global
