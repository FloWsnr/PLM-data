"""Darcy flow equation for porous media."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("darcy")
class DarcyPDE(ScalarPDEPreset):
    """Darcy flow equation for porous media.

    The Darcy flow equation describes pressure-driven flow through porous media,
    derived from Darcy's law combined with mass conservation:

        dp/dt = K * laplace(p) + f

    where:
        - p is the pressure field
        - K is the permeability/hydraulic conductivity
        - f is the source/sink term (positive = source, negative = sink)

    This equation is fundamental in:
        - Groundwater hydrology and aquifer modeling
        - Oil and gas reservoir simulation
        - Filtration and separation processes
        - Soil mechanics and geotechnical engineering

    The equation reduces to the standard diffusion equation when f = 0.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="darcy",
            category="fluids",
            description="Darcy flow equation for porous media",
            equations={"p": "K * laplace(p) + f"},
            parameters=[
                PDEParameter("K", "Permeability/hydraulic conductivity"),
                PDEParameter("f", "Source/sink term (positive = source)"),
            ],
            num_fields=1,
            field_names=["p"],
            reference="https://en.wikipedia.org/wiki/Darcy%27s_law",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: Any,
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Darcy flow PDE.

        Args:
            parameters: Dictionary containing 'K' (permeability) and 'f' (source term).
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        K = parameters["K"]
        f = parameters["f"]

        bc_spec = self._convert_bc(bc, grid.dim)

        return PDE(
            rhs={"p": f"{K} * laplace(p) + {f}"},
            bc=bc_spec,
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        K = parameters["K"]
        f = parameters["f"]
        return {"p": f"{K} * laplace(p) + {f}"}
