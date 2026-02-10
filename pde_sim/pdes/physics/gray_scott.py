"""Gray-Scott reaction-diffusion system."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("gray-scott")
class GrayScottPDE(MultiFieldPDEPreset):
    """Gray-Scott reaction-diffusion system.

    Based on the visualpde.com formulation:

        du/dt = laplace(u) + u^2*v - (a+b)*u
        dv/dt = D * laplace(v) - u^2*v + a*(1-v)

    where:
        - u is the activator concentration (autocatalytic species)
        - v is the substrate concentration (fuel species)
        - a is the feed rate
        - b is the kill/removal rate
        - D is the diffusion ratio (v diffuses D times faster than u)

    The system exhibits an extraordinary range of patterns depending on (a,b):
        - Labyrinthine/stripe patterns
        - Stationary and pulsating spots
        - Self-replicating spots
        - Worm-like structures
        - Holes, spatiotemporal chaos
        - Moving spots (gliders)

    Reference: Pearson (1993) "Complex Patterns in a Simple System"
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="gray-scott",
            category="physics",
            description="Gray-Scott reaction-diffusion pattern formation",
            equations={
                "u": "laplace(u) + u**2 * v - (a + b) * u",
                "v": "D * laplace(v) - u**2 * v + a * (1 - v)",
            },
            parameters=[
                PDEParameter("a", "Feed rate"),
                PDEParameter("b", "Kill/removal rate"),
                PDEParameter("D", "Diffusion ratio (v diffuses D times faster than u)"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Pearson (1993) Complex patterns in a simple system",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Gray-Scott PDE system.

        Args:
            parameters: Dictionary with a, b, D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        a = parameters.get("a", 0.037)
        b = parameters.get("b", 0.06)
        D = parameters.get("D", 2.0)

        # Gray-Scott equations from reference:
        # du/dt = laplace(u) + u^2*v - (a+b)*u
        # dv/dt = D*laplace(v) - u^2*v + a*(1-v)
        return PDE(
            rhs={
                "u": f"laplace(u) + u**2 * v - ({a} + {b}) * u",
                "v": f"{D} * laplace(v) - u**2 * v + {a} * (1 - v)",
            },
            bc=self._convert_bc(bc),
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        a = parameters.get("a", 0.037)
        b = parameters.get("b", 0.06)
        D = parameters.get("D", 2.0)

        return {
            "u": f"laplace(u) + u**2 * v - ({a} + {b}) * u",
            "v": f"{D} * laplace(v) - u**2 * v + {a} * (1 - v)",
        }
