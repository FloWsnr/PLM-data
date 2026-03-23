"""Poisson equation preset: -div(kappa * grad(u)) = f on a rectangle."""

import ufl
from dolfinx import fem, mesh as dmesh
from petsc4py.PETSc import ScalarType

from plm_data.core.config import SimulationConfig
from plm_data.presets import register_preset
from plm_data.presets.base import SteadyLinearPreset
from plm_data.presets.metadata import PDEMetadata, PDEParameter


@register_preset("poisson")
class PoissonPreset(SteadyLinearPreset):

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="poisson",
            category="basic",
            description="Poisson equation -div(kappa * grad(u)) = f",
            equations={"u": "-∇·(κ ∇u) = f"},
            parameters=[
                PDEParameter("kappa", "Diffusion coefficient"),
                PDEParameter("f_amplitude", "Source term amplitude"),
            ],
            field_names=["u"],
            steady_state=True,
            supported_dimensions=[2],
            recommended_config={
                "preset": "poisson",
                "parameters": {"kappa": 1.0, "f_amplitude": 1.0},
                "solver": {"ksp_type": "preonly", "pc_type": "lu"},
                "domain": {
                    "type": "rectangle",
                    "size": [1.0, 1.0],
                    "mesh_resolution": [64, 64],
                },
                "output_resolution": [64, 64],
                "output": {
                    "path": "./output",
                    "num_frames": 1,
                    "formats": ["numpy"],
                },
                "seed": 42,
            },
        )

    def create_function_space(self, msh, config):
        return fem.functionspace(msh, ("Lagrange", 2))

    def create_boundary_conditions(self, V, msh, config):
        tdim = msh.topology.dim
        msh.topology.create_connectivity(tdim - 1, tdim)
        facets = dmesh.exterior_facet_indices(msh.topology)
        dofs = fem.locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)
        bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
        return [bc]

    def create_forms(self, V, msh, config):
        kappa = config.parameters["kappa"]
        f_amp = config.parameters["f_amplitude"]

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)

        # Source term: f = f_amp * sin(pi*x) * sin(pi*y)
        f = f_amp * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

        a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx
        return a, L
