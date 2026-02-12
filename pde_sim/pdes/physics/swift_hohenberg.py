"""Swift-Hohenberg equation for pattern formation."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("swift-hohenberg")
class SwiftHohenbergPDE(ScalarPDEPreset):
    """Swift-Hohenberg equation for pattern formation.

    Based on visualpde.com formulation:

        du/dt = r*u - (k_c^2 + laplace)^2*u + a*u^2 + b*u^3 + c*u^5

    Expanding (k_c^2 + laplace)^2*u = k_c^4*u + 2*k_c^2*laplace(u) + laplace(laplace(u))

    So: du/dt = r*u - k_c^4*u - 2*k_c^2*laplace(u) - laplace(laplace(u)) + a*u^2 + b*u^3 + c*u^5

    With k_c = 1 (default):
        du/dt = (r-1)*u - 2*laplace(u) - laplace(laplace(u)) + a*u^2 + b*u^3 + c*u^5

    Key features:
        - Pattern selection: stripes, hexagons, squares depending on parameters
        - Subcriticality: when r < 0, a > 0, b < 0, both patterned and uniform states stable
        - Localised solutions: stable spots or patches of stripes
        - Homoclinic snaking: infinitely many coexisting localised states

    Stability requirement: c < 0 (or b < 0 if c = 0) for bounded solutions.

    Reference: Swift & Hohenberg (1977), Burke & Knobloch (2006)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="swift-hohenberg",
            category="physics",
            description="Swift-Hohenberg pattern formation near criticality",
            equations={
                "u": "r*u - (k_c**2 + laplace)**2*u + a*u**2 + b*u**3 + c*u**5",
            },
            parameters=[
                PDEParameter("r", "Bifurcation parameter (distance from onset)"),
                PDEParameter("a", "Quadratic coefficient (enables hexagons)"),
                PDEParameter("b", "Cubic coefficient (saturation)"),
                PDEParameter("c", "Quintic coefficient (must be negative for stability)"),
                PDEParameter("D", "Effective diffusion scale (k_c = 1)"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Swift & Hohenberg (1977) Hydrodynamic fluctuations",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Swift-Hohenberg PDE.

        Args:
            parameters: Dictionary with r, a, b, c, D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        r = parameters["r"]
        a = parameters["a"]
        b = parameters["b"]
        c = parameters["c"]
        D = parameters["D"]

        # k_c = 1 (critical wavenumber)
        k_c = 1.0
        k_sq = k_c * k_c

        # Swift-Hohenberg: du/dt = r*u - (k_c^2 + laplace)^2*u + a*u^2 + b*u^3 + c*u^5
        # Expanding: (k_c^2 + laplace)^2*u = k_c^4*u + 2*k_c^2*laplace(u) + laplace(laplace(u))
        # With D scaling the laplacians
        rhs = f"({r} - {k_sq**2}) * u - 2 * {k_sq} * {D} * laplace(u) - {D**2} * laplace(laplace(u))"
        if a != 0:
            rhs += f" + {a} * u**2"
        if b != 0:
            rhs += f" + {b} * u**3"
        if c != 0:
            rhs += f" + {c} * u**5"

        return PDE(
            rhs={"u": rhs},
            bc=self._convert_bc(bc),
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        r = parameters["r"]
        a = parameters["a"]
        b = parameters["b"]
        c = parameters["c"]

        return {
            "u": f"{r}*u - (1 + laplace)**2*u + {a}*u**2 + {b}*u**3 + {c}*u**5",
        }
