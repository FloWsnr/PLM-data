"""Superlattice pattern formation via coupled reaction-diffusion systems."""

from typing import Any

from pde import PDE, CartesianGrid

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
                PDEParameter("a", "Brusselator feed parameter"),
                PDEParameter("b", "Brusselator removal parameter"),
                PDEParameter("c", "Lengyel-Epstein feed parameter"),
                PDEParameter("d", "Lengyel-Epstein rate parameter"),
                PDEParameter("alpha", "Coupling strength between subsystems"),
                PDEParameter("D_uone", "Diffusion of u1 (Brusselator activator)"),
                PDEParameter("D_utwo", "Diffusion of v1 (Brusselator inhibitor)"),
                PDEParameter("D_uthree", "Diffusion of u2 (Lengyel-Epstein activator)"),
                PDEParameter("D_ufour", "Diffusion of v2 (Lengyel-Epstein inhibitor)"),
            ],
            num_fields=4,
            field_names=["u1", "v1", "u2", "v2"],
            reference="De Wit et al. (1999) Simple and superlattice Turing patterns",
            supported_dimensions=[1, 2, 3],
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
        a = parameters["a"]
        b = parameters["b"]
        c = parameters["c"]
        d = parameters["d"]
        alpha = parameters["alpha"]
        D_uone = parameters["D_uone"]
        D_utwo = parameters["D_utwo"]
        D_uthree = parameters["D_uthree"]
        D_ufour = parameters["D_ufour"]

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

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        a = parameters["a"]
        b = parameters["b"]
        c = parameters["c"]
        d = parameters["d"]
        alpha = parameters["alpha"]
        D_uone = parameters["D_uone"]
        D_utwo = parameters["D_utwo"]
        D_uthree = parameters["D_uthree"]
        D_ufour = parameters["D_ufour"]

        return {
            "u1": f"{D_uone}*laplace(u1) + {a} - ({b}+1)*u1 + u1**2*v1 + {alpha}*u1*u2*(u2-u1)",
            "v1": f"{D_utwo}*laplace(v1) + {b}*u1 - u1**2*v1",
            "u2": f"{D_uthree}*laplace(u2) + {c} - u2 - 4*u2*v2/(1+u2**2) + {alpha}*u1*u2*(u1-u2)",
            "v2": f"{D_ufour}*laplace(v2) + {d}*(u2 - u2*v2/(1+u2**2))",
        }
