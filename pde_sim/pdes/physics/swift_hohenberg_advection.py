"""Swift-Hohenberg equation with advection."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("swift-hohenberg-advection")
class SwiftHohenbergAdvectionPDE(ScalarPDEPreset):
    """Swift-Hohenberg equation with advection (flow).

    Based on visualpde.com formulation for unidirectional advection:

        du/dt = r*u - (1 + laplace)^2*u + a*u^2 + b*u^3 + V*(cos(theta)*d_dx(u) + sin(theta)*d_dy(u))

    This extends the Swift-Hohenberg equation with advection terms,
    creating a system where patterns can both form and move.

    Key phenomena:
        - Translate: patterns move in a fixed direction while maintaining shape
        - Rotate: patterns spin around a center point (with rotational advection)
        - Deform: patterns change shape due to flow gradients
        - Destabilize: patterns break apart if flow is too strong

    Reference: Cross & Hohenberg (1993), Knobloch (2015)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="swift-hohenberg-advection",
            category="physics",
            description="Swift-Hohenberg with advection for moving patterns",
            equations={
                "u": "r*u - (1 + laplace)**2*u + a*u**2 + b*u**3 + V*(cos(theta)*d_dx(u) + sin(theta)*d_dy(u))",
            },
            parameters=[
                PDEParameter("r", "Bifurcation parameter (subcritical for localised)"),
                PDEParameter("a", "Quadratic coefficient"),
                PDEParameter("b", "Cubic coefficient"),
                PDEParameter("c", "Quintic coefficient"),
                PDEParameter("V", "Advection velocity magnitude"),
                PDEParameter("theta", "Advection direction angle (radians)"),
                PDEParameter("D", "Diffusion scale"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Cross & Hohenberg (1993) Pattern formation outside equilibrium",
            supported_dimensions=[2],  # Currently 2D only (uses theta angle for 2D advection)
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Swift-Hohenberg with advection PDE.

        Args:
            parameters: Dictionary with r, a, b, c, V, theta, D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        r = parameters.get("r", -0.28)
        a = parameters.get("a", 1.6)
        b = parameters.get("b", -1.0)
        c = parameters.get("c", -1.0)
        V = parameters.get("V", 2.0)
        theta = parameters.get("theta", -2.0)
        D = parameters.get("D", 1.0)

        # k_c = 1 (critical wavenumber)
        k_sq = 1.0

        # Swift-Hohenberg with advection:
        # du/dt = r*u - (1 + laplace)^2*u + a*u^2 + b*u^3 + c*u^5
        #       + V*(cos(theta)*d_dx(u) + sin(theta)*d_dy(u))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rhs = f"({r} - {k_sq**2}) * u - 2 * {k_sq} * {D} * laplace(u) - {D**2} * laplace(laplace(u))"
        if a != 0:
            rhs += f" + {a} * u**2"
        if b != 0:
            rhs += f" + {b} * u**3"
        if c != 0:
            rhs += f" + {c} * u**5"

        # Advection terms
        if V != 0:
            if cos_theta != 0:
                rhs += f" + {V * cos_theta} * d_dx(u)"
            if sin_theta != 0:
                rhs += f" + {V * sin_theta} * d_dy(u)"

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
        """Create initial state for Swift-Hohenberg with advection.

        Default: localised structure with specific symmetry.
        """
        if ic_type in ("swift-hohenberg-advection-default", "default"):
            amplitude = ic_params.get("amplitude", 0.5)
            P = ic_params.get("P", 3)  # Pattern symmetry: 1=D4, 2=D6, 3=D12
            rng = np.random.default_rng(ic_params.get("seed"))

            # Get domain info
            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            # Create coordinates centered at domain center
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Pattern center (randomize if not specified)
            cx = ic_params.get("cx")
            cy = ic_params.get("cy")
            if cx is None or cx == "random" or cy is None or cy == "random":
                raise ValueError("swift-hohenberg-advection requires cx and cy (or random)")
            cx = x_bounds[0] + cx * Lx
            cy = y_bounds[0] + cy * Ly
            X_c = X - cx
            Y_c = Y - cy

            # Localized Gaussian envelope
            sigma = 0.15 * min(Lx, Ly)
            envelope = np.exp(-(X_c**2 + Y_c**2) / (2 * sigma**2))

            # Pattern with specified symmetry
            R = np.sqrt(X_c**2 + Y_c**2)
            theta_local = np.arctan2(Y_c, X_c)

            # Number of lobes based on P
            n_lobes = {1: 4, 2: 6, 3: 12}.get(int(P), 4)
            pattern = np.cos(n_lobes * theta_local) * np.cos(R / sigma * np.pi)

            data = amplitude * envelope * pattern

            # Add small noise
            data += 0.01 * rng.standard_normal(grid.shape)

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("swift-hohenberg-advection-default", "default"):
            resolved = ic_params.copy()
            if "cx" not in resolved or "cy" not in resolved:
                raise ValueError("swift-hohenberg-advection requires cx and cy (or random)")
            if resolved["cx"] == "random" or resolved["cy"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["cx"] == "random":
                    resolved["cx"] = rng.uniform(0.2, 0.8)
                if resolved["cy"] == "random":
                    resolved["cy"] = rng.uniform(0.2, 0.8)
            if resolved["cx"] is None or resolved["cy"] is None:
                raise ValueError("swift-hohenberg-advection requires cx and cy (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        r = parameters.get("r", -0.28)
        a = parameters.get("a", 1.6)
        b = parameters.get("b", -1.0)
        V = parameters.get("V", 2.0)
        theta = parameters.get("theta", -2.0)

        return {
            "u": f"{r}*u - (1 + laplace)**2*u + {a}*u**2 + {b}*u**3 + {V}*(cos({theta})*d_dx(u) + sin({theta})*d_dy(u))",
        }
