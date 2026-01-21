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
                "u": "laplace(u) + a + A*X + u**2/v - (b + B*X)*u",
                "v": "D * laplace(v) + u**2 - c*v",
                "X": "0 (static coordinate field)",
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
            num_fields=3,
            field_names=["u", "v", "X"],
            reference="Iron, Ward & Wei (2001)",
            supported_dimensions=[1, 2, 3],
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

        # G(x) = A*X and H(x) = B*X where X is a static field containing x/L
        # X field is set up in create_initial_state with values in [0, 1]
        if A != 0 or B != 0:
            # With heterogeneity - X is the normalized coordinate field
            u_rhs = f"laplace(u) + {a} + {A}*X + u**2 / (v + 1e-10) - ({b} + {B}*X) * u"
        else:
            # Homogeneous case
            u_rhs = f"laplace(u) + {a} + u**2 / (v + 1e-10) - {b} * u"

        v_rhs = f"{D} * laplace(v) + u**2 - {c} * v"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": v_rhs,
                "X": "0",  # Static coordinate field
            },
            **self._get_pde_bc_kwargs(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state near steady state with perturbation.

        Includes a static X field containing normalized x-coordinates (x/L)
        for implementing spatial heterogeneity.
        """
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

        # Create normalized x-coordinate field X = x/L (values in [0, 1])
        x_bounds = grid.axes_bounds[0]
        L = x_bounds[1] - x_bounds[0]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        # Normalize to [0, 1]
        x_normalized = (x - x_bounds[0]) / L
        # Broadcast to 2D grid (x varies along first axis)
        X_data = np.broadcast_to(x_normalized[:, np.newaxis], grid.shape).copy()

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        X = ScalarField(grid, X_data)
        X.label = "X"

        return FieldCollection([u, v, X])
