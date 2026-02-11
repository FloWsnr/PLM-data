"""Phase-Field Crystal model for crystallization dynamics."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("phase-field-crystal")
class PhaseFieldCrystalPDE(ScalarPDEPreset):
    """Phase-Field Crystal model for crystallization dynamics.

    ∂φ/∂t = Δ[(1+Δ)²φ + φ³ − εφ]

    Expanded: ∂φ/∂t = (1−ε)Δφ + 2Δ²φ + Δ³φ + Δ(φ³)

    Models crystallization, grain boundary dynamics, and elastic interactions
    at atomic length scales but diffusive time scales. The 6th-order conserved
    dynamics naturally produce periodic lattice structures.

    Reference: Elder et al. (2002) "Modeling Elasticity in Crystal Growth"
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="phase-field-crystal",
            category="physics",
            description="6th-order conserved dynamics for crystallization",
            equations={
                "u": "(1-epsilon)*laplace(u) + 2*laplace(laplace(u)) + laplace(laplace(laplace(u))) + laplace(u**3)",
            },
            parameters=[
                PDEParameter("epsilon", "Undercooling parameter controlling crystal stability"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Elder et al. (2002) Modeling Elasticity in Crystal Growth",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        epsilon = parameters["epsilon"]

        coeff = 1 - epsilon
        rhs = (
            f"{coeff}*laplace(u) + 2*laplace(laplace(u)) "
            f"+ laplace(laplace(laplace(u))) + laplace(u**3)"
        )

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
        """Create initial state for Phase-Field Crystal.

        Default: Small random perturbations around mean density to trigger crystallization.
        """
        if ic_type in ("phase-field-crystal-default", "default"):
            mean_density = ic_params["mean_density"]
            amplitude = ic_params["amplitude"]
            rng = np.random.default_rng(ic_params.get("seed"))

            data = mean_density + amplitude * rng.standard_normal(grid.shape)
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        epsilon = parameters["epsilon"]
        return {
            "u": f"({1 - epsilon})*laplace(u) + 2*laplace(laplace(u)) + laplace(laplace(laplace(u))) + laplace(u**3)",
        }
