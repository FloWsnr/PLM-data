"""Superlattice pattern formation via coupled reaction-diffusion systems."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("superlattice")
class SuperlatticePDE(MultiFieldPDEPreset):
    """Superlattice pattern formation via coupled Brusselator + Lengyll-Epstein.

    Based on visualpde.com formulation - coupling two pattern-forming systems:

    Brusselator subsystem (u1, v1):
        du1/dt = D_u1·∇²u1 + a - (b+1)u1 + u1²v1 + α·u1·u2·(u2-u1)
        dv1/dt = D_v1·∇²v1 + b·u1 - u1²v1

    Lengyll-Epstein subsystem (u2, v2):
        du2/dt = D_u2·∇²u2 + c - u2 - 4u2v2/(1+u2²) + α·u1·u2·(u1-u2)
        dv2/dt = D_v2·∇²v2 + d·[u2 - u2v2/(1+u2²)]

    The coupling parameter α controls interaction between subsystems.
    Different diffusion ratios create overlapping pattern wavelengths,
    leading to superlattice structures.

    Reference: https://visualpde.com/nonlinear-physics/superlattice-patterns
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="superlattice",
            category="physics",
            description="Coupled Brusselator + Lengyll-Epstein superlattice",
            equations={
                "u1": "D_u1*laplace(u1) + a - (b+1)*u1 + u1^2*v1 + alpha*u1*u2*(u2-u1)",
                "v1": "D_v1*laplace(v1) + b*u1 - u1^2*v1",
                "u2": "D_u2*laplace(u2) + c - u2 - 4*u2*v2/(1+u2^2) + alpha*u1*u2*(u1-u2)",
                "v2": "D_v2*laplace(v2) + d*(u2 - u2*v2/(1+u2^2))",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=3.0,
                    description="Brusselator parameter a",
                    min_value=1.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="b",
                    default=9.0,
                    description="Brusselator parameter b",
                    min_value=5.0,
                    max_value=15.0,
                ),
                PDEParameter(
                    name="c",
                    default=15.0,
                    description="Lengyll-Epstein parameter c",
                    min_value=5.0,
                    max_value=25.0,
                ),
                PDEParameter(
                    name="d",
                    default=9.0,
                    description="Lengyll-Epstein parameter d",
                    min_value=5.0,
                    max_value=15.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=0.15,
                    description="Coupling parameter",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="D_u1",
                    default=3.1,
                    description="Diffusion of u1 (Brusselator activator)",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D_v1",
                    default=13.95,
                    description="Diffusion of v1 (Brusselator inhibitor)",
                    min_value=1.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="D_u2",
                    default=18.9,
                    description="Diffusion of u2 (Lengyll-Epstein activator)",
                    min_value=1.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="D_v2",
                    default=670.0,
                    description="Diffusion of v2 (Lengyll-Epstein inhibitor)",
                    min_value=100.0,
                    max_value=1000.0,
                ),
            ],
            num_fields=4,
            field_names=["u1", "v1", "u2", "v2"],
            reference="https://visualpde.com/nonlinear-physics/superlattice-patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 3.0)
        b = parameters.get("b", 9.0)
        c = parameters.get("c", 15.0)
        d = parameters.get("d", 9.0)
        alpha = parameters.get("alpha", 0.15)
        D_u1 = parameters.get("D_u1", 3.1)
        D_v1 = parameters.get("D_v1", 13.95)
        D_u2 = parameters.get("D_u2", 18.9)
        D_v2 = parameters.get("D_v2", 670.0)

        # Coupled Brusselator + Lengyll-Epstein system
        return PDE(
            rhs={
                "u1": f"{D_u1}*laplace(u1) + {a} - ({b}+1)*u1 + u1**2*v1 + {alpha}*u1*u2*(u2-u1)",
                "v1": f"{D_v1}*laplace(v1) + {b}*u1 - u1**2*v1",
                "u2": f"{D_u2}*laplace(u2) + {c} - u2 - 4*u2*v2/(1+u2**2) + {alpha}*u1*u2*(u1-u2)",
                "v2": f"{D_v2}*laplace(v2) + {d}*(u2 - u2*v2/(1+u2**2))",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near equilibrium with perturbations."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        # Get parameters for equilibrium calculation
        a = ic_params.get("a", 3.0)
        b = ic_params.get("b", 9.0)
        c = ic_params.get("c", 15.0)
        d = ic_params.get("d", 9.0)

        # Approximate equilibrium values (simplified)
        # Brusselator equilibrium: u1* = a, v1* = b/a
        u1_eq = a
        v1_eq = b / a

        # Lengyll-Epstein equilibrium (approximate)
        u2_eq = c / (1 + 4 / (1 + c**2))  # Simplified
        v2_eq = 1.0  # Approximate

        # Add noise around equilibrium
        u1_data = u1_eq + noise * np.random.randn(*grid.shape)
        v1_data = v1_eq + noise * np.random.randn(*grid.shape)
        u2_data = u2_eq + noise * np.random.randn(*grid.shape)
        v2_data = v2_eq + noise * np.random.randn(*grid.shape)

        # Ensure positive values
        u1_data = np.maximum(u1_data, 0.01)
        v1_data = np.maximum(v1_data, 0.01)
        u2_data = np.maximum(u2_data, 0.01)
        v2_data = np.maximum(v2_data, 0.01)

        u1 = ScalarField(grid, u1_data)
        u1.label = "u1"
        v1 = ScalarField(grid, v1_data)
        v1.label = "v1"
        u2 = ScalarField(grid, u2_data)
        u2.label = "u2"
        v2 = ScalarField(grid, v2_data)
        v2.label = "v2"

        return FieldCollection([u1, v1, u2, v2])
