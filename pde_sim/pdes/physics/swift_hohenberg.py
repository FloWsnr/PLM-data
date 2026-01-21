"""Swift-Hohenberg equation for pattern formation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

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
                PDEParameter(
                    name="r",
                    default=0.1,
                    description="Bifurcation parameter (distance from onset)",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Quadratic coefficient (enables hexagons)",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="b",
                    default=1.0,
                    description="Cubic coefficient (saturation)",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="c",
                    default=-1.0,
                    description="Quintic coefficient (must be negative for stability)",
                    min_value=-5.0,
                    max_value=0.0,
                ),
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Effective diffusion scale (k_c = 1)",
                    min_value=0.1,
                    max_value=5.0,
                ),
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
        r = parameters.get("r", 0.1)
        a = parameters.get("a", 1.0)
        b = parameters.get("b", 1.0)
        c = parameters.get("c", -1.0)
        D = parameters.get("D", 1.0)

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

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for Swift-Hohenberg.

        Default: u = 0 with small perturbations.
        """
        if ic_type in ("swift-hohenberg-default", "default"):
            amplitude = ic_params.get("amplitude", 0.1)
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)
            data = amplitude * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        r = parameters.get("r", 0.1)
        a = parameters.get("a", 1.0)
        b = parameters.get("b", 1.0)
        c = parameters.get("c", -1.0)

        return {
            "u": f"{r}*u - (1 + laplace)**2*u + {a}*u**2 + {b}*u**3 + {c}*u**5",
        }
