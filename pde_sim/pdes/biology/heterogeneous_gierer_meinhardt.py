"""Heterogeneous Gierer-Meinhardt model with spatial parameter gradients."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("heterogeneous-gierer-meinhardt")
class HeterogeneousGiererMeinhardtPDE(MultiFieldPDEPreset):
    """Heterogeneous Gierer-Meinhardt system with spatial gradients.

    Extended Gierer-Meinhardt with position-dependent parameters:

        du/dt = laplace(u) + a + G(x) + u^2/v - [b + H(x)]*u
        dv/dt = D * laplace(v) + u^2 - c*v

    where:
        G(x) = A*x/L  (production gradient)
        H(x) = B*x/L  (decay gradient)

    Key phenomena:
        - Localized patterns: form preferentially in certain regions
        - Spike pinning: spikes localize at specific positions
        - Spike motion: spikes migrate along parameter gradients
        - Hopf instabilities: local oscillations in spike amplitude

    References:
        Iron, Ward & Wei (2001). Physica D, 150(1-2), 25-62.
        Ward & Wei (2003). J. Nonlinear Science, 13(2), 209-264.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heterogeneous-gierer-meinhardt",
            category="biology",
            description="Heterogeneous Gierer-Meinhardt with spatial gradients",
            equations={
                "u": "laplace(u) + a + G(x) + u**2/v - [b + H(x)]*u",
                "v": "D * laplace(v) + u**2 - c*v",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=55.0,
                    description="Inhibitor diffusion ratio",
                    min_value=1.0,
                    max_value=200.0,
                ),
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Base activator production",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="b",
                    default=1.5,
                    description="Base activator decay rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="c",
                    default=6.1,
                    description="Inhibitor decay rate",
                    min_value=0.1,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="A",
                    default=0.0,
                    description="Production gradient strength",
                    min_value=-1.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="B",
                    default=0.0,
                    description="Decay gradient strength",
                    min_value=0.0,
                    max_value=5.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Iron, Ward & Wei (2001)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 55.0)
        a = parameters.get("a", 1.0)
        b = parameters.get("b", 1.5)
        c = parameters.get("c", 6.1)
        A = parameters.get("A", 0.0)
        B = parameters.get("B", 0.0)

        # Get domain length
        L = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]

        # G(x) = A*x/L and H(x) = B*x/L
        # x is the first coordinate accessed via x[0]
        # Note: py-pde uses x to refer to coordinates. For 2D, x[0] is the first axis.
        if A != 0 or B != 0:
            # With heterogeneity
            u_rhs = f"laplace(u) + {a} + {A}*x[0]/{L} + u**2 / (v + 1e-10) - ({b} + {B}*x[0]/{L}) * u"
        else:
            # Homogeneous case
            u_rhs = f"laplace(u) + {a} + u**2 / (v + 1e-10) - {b} * u"

        v_rhs = f"{D} * laplace(v) + u**2 - {c} * v"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": v_rhs,
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state near steady state with perturbation."""
        noise = ic_params.get("noise", 0.01)

        # Initial values near steady state (approximate)
        u0 = ic_params.get("u0", 1.0)
        v0 = ic_params.get("v0", 1.0)

        np.random.seed(ic_params.get("seed"))
        u_data = u0 * (1 + noise * np.random.randn(*grid.shape))
        v_data = v0 * (1 + noise * np.random.randn(*grid.shape))

        # Ensure positive values
        u_data = np.maximum(u_data, 0.01)
        v_data = np.maximum(v_data, 0.01)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])
