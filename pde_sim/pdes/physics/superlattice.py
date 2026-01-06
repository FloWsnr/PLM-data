"""Superlattice pattern formation via coupled reaction-diffusion systems."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("superlattice")
class SuperlatticePDE(MultiFieldPDEPreset):
    """Superlattice pattern formation via coupled Brusselator + Lengyel-Epstein.

    Based on visualpde.com formulation - coupling two pattern-forming systems:

    Brusselator subsystem (u1, v1):
        du1/dt = D_uone * laplace(u1) + a - (b+1)*u1 + u1^2*v1 + alpha*u1*u2*(u2-u1)
        dv1/dt = D_utwo * laplace(v1) + b*u1 - u1^2*v1

    Lengyel-Epstein subsystem (u2, v2):
        du2/dt = D_uthree * laplace(u2) + c - u2 - 4*u2*v2/(1+u2^2) + alpha*u1*u2*(u1-u2)
        dv2/dt = D_ufour * laplace(v2) + d*(u2 - u2*v2/(1+u2^2))

    Complex spatial patterns arise from the coupling of two oscillatory
    pattern-forming systems with different intrinsic wavelengths.

    Key phenomena:
        - Simple patterns: when modes reinforce each other
        - Superlattice patterns: when spatial resonance conditions are met
        - Superposition patterns: when modes don't resonate but coexist
        - Spatiotemporal chaos: when modes compete destructively

    Reference: De Wit et al. (1999), Yang et al. (2004)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="superlattice",
            category="physics",
            description="Coupled Brusselator + Lengyel-Epstein superlattice patterns",
            equations={
                "u1": "D_uone*laplace(u1) + a - (b+1)*u1 + u1**2*v1 + alpha*u1*u2*(u2-u1)",
                "v1": "D_utwo*laplace(v1) + b*u1 - u1**2*v1",
                "u2": "D_uthree*laplace(u2) + c - u2 - 4*u2*v2/(1+u2**2) + alpha*u1*u2*(u1-u2)",
                "v2": "D_ufour*laplace(v2) + d*(u2 - u2*v2/(1+u2**2))",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=3.0,
                    description="Brusselator feed parameter",
                    min_value=1.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="b",
                    default=9.0,
                    description="Brusselator removal parameter",
                    min_value=5.0,
                    max_value=15.0,
                ),
                PDEParameter(
                    name="c",
                    default=15.0,
                    description="Lengyel-Epstein feed parameter",
                    min_value=5.0,
                    max_value=25.0,
                ),
                PDEParameter(
                    name="d",
                    default=9.0,
                    description="Lengyel-Epstein rate parameter",
                    min_value=5.0,
                    max_value=15.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=0.15,
                    description="Coupling strength between subsystems",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="D_uone",
                    default=4.3,
                    description="Diffusion of u1 (Brusselator activator)",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D_utwo",
                    default=50.0,
                    description="Diffusion of v1 (Brusselator inhibitor)",
                    min_value=1.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="D_uthree",
                    default=22.0,
                    description="Diffusion of u2 (Lengyel-Epstein activator)",
                    min_value=1.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="D_ufour",
                    default=660.0,
                    description="Diffusion of v2 (Lengyel-Epstein inhibitor)",
                    min_value=100.0,
                    max_value=1000.0,
                ),
            ],
            num_fields=4,
            field_names=["u1", "v1", "u2", "v2"],
            reference="De Wit et al. (1999) Simple and superlattice Turing patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Superlattice PDE system.

        Args:
            parameters: Dictionary with a, b, c, d, alpha, D_uone, D_utwo, D_uthree, D_ufour.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        a = parameters.get("a", 3.0)
        b = parameters.get("b", 9.0)
        c = parameters.get("c", 15.0)
        d = parameters.get("d", 9.0)
        alpha = parameters.get("alpha", 0.15)
        D_uone = parameters.get("D_uone", 4.3)
        D_utwo = parameters.get("D_utwo", 50.0)
        D_uthree = parameters.get("D_uthree", 22.0)
        D_ufour = parameters.get("D_ufour", 660.0)

        # Coupled Brusselator + Lengyel-Epstein system
        return PDE(
            rhs={
                "u1": f"{D_uone}*laplace(u1) + {a} - ({b}+1)*u1 + u1**2*v1 + {alpha}*u1*u2*(u2-u1)",
                "v1": f"{D_utwo}*laplace(v1) + {b}*u1 - u1**2*v1",
                "u2": f"{D_uthree}*laplace(u2) + {c} - u2 - 4*u2*v2/(1+u2**2) + {alpha}*u1*u2*(u1-u2)",
                "v2": f"{D_ufour}*laplace(v2) + {d}*(u2 - u2*v2/(1+u2**2))",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near equilibrium with perturbations.

        Default: u1 = a (Brusselator equilibrium) with small random perturbations.
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        noise = ic_params.get("noise", 0.1)

        # Get parameters for equilibrium calculation
        a = ic_params.get("a", 3.0)
        b = ic_params.get("b", 9.0)
        c = ic_params.get("c", 15.0)

        # Brusselator equilibrium: u1* = a, v1* = b/a
        u1_eq = a
        v1_eq = b / a

        # Lengyel-Epstein equilibrium (approximate)
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

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        a = parameters.get("a", 3.0)
        b = parameters.get("b", 9.0)
        c = parameters.get("c", 15.0)
        d = parameters.get("d", 9.0)
        alpha = parameters.get("alpha", 0.15)
        D_uone = parameters.get("D_uone", 4.3)
        D_utwo = parameters.get("D_utwo", 50.0)
        D_uthree = parameters.get("D_uthree", 22.0)
        D_ufour = parameters.get("D_ufour", 660.0)

        return {
            "u1": f"{D_uone}*laplace(u1) + {a} - ({b}+1)*u1 + u1**2*v1 + {alpha}*u1*u2*(u2-u1)",
            "v1": f"{D_utwo}*laplace(v1) + {b}*u1 - u1**2*v1",
            "u2": f"{D_uthree}*laplace(u2) + {c} - u2 - 4*u2*v2/(1+u2**2) + {alpha}*u1*u2*(u1-u2)",
            "v2": f"{D_ufour}*laplace(v2) + {d}*(u2 - u2*v2/(1+u2**2))",
        }
