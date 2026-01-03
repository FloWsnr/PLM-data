"""Heat/diffusion equation."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("heat")
class HeatPDE(ScalarPDEPreset):
    """Heat/diffusion equation.

    The heat equation describes the diffusion of heat (or any quantity
    that diffuses) over time:

        dT/dt = D * laplace(T)

    where T is the temperature (or concentration) and D is the diffusion
    coefficient.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heat",
            category="basic",
            description="Heat diffusion equation",
            equations={"T": "D * laplace(T)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=0.5,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="Basic parabolic PDE",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the heat equation PDE.

        Args:
            parameters: Dictionary containing 'D' (diffusion coefficient).
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 1.0)

        # Convert BC to py-pde format
        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={"T": f"{D} * laplace(T)"},
            bc=bc_spec,
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 1.0)
        return {"T": f"{D} * laplace(T)"}


@register_pde("inhomogeneous-heat")
class InhomogeneousHeatPDE(ScalarPDEPreset):
    """Inhomogeneous heat equation with spatially varying diffusion.

    dT/dt = div(g(x,y) * grad(T)) + f(x,y)

    This simplified version uses constant coefficients that can vary:
        dT/dt = D * laplace(T) + source

    where source is a constant source/sink term.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="inhomogeneous-heat",
            category="basic",
            description="Heat equation with source term",
            equations={"T": "D * laplace(T) + source"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="source",
                    default=0.0,
                    description="Constant source/sink term",
                    min_value=-10.0,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="Inhomogeneous parabolic PDE",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        source = parameters.get("source", 0.0)

        # Build equation string
        if source == 0:
            rhs = f"{D} * laplace(T)"
        elif source > 0:
            rhs = f"{D} * laplace(T) + {source}"
        else:
            rhs = f"{D} * laplace(T) - {abs(source)}"

        return PDE(
            rhs={"T": rhs},
            bc=self._convert_bc(bc),
        )
