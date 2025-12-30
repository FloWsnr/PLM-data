"""Kuramoto-Sivashinsky equation for chaotic pattern formation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("kuramoto-sivashinsky")
class KuramotoSivashinskyPDE(ScalarPDEPreset):
    """Kuramoto-Sivashinsky equation.

    A canonical model for spatiotemporal chaos:

        du/dt = -laplace(u) - nu * laplace(laplace(u)) - 0.5 * |grad(u)|^2

    or equivalently:

        du/dt = -laplace(u) - nu * laplace(laplace(u)) - 0.5 * gradient_squared(u)

    where:
        - u is the field variable (e.g., flame front position)
        - nu is the fourth-order diffusion coefficient

    This equation exhibits:
        - Linear instability at intermediate wavelengths
        - Nonlinear saturation
        - Spatiotemporal chaos

    Reference: Kuramoto (1978), Sivashinsky (1977)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kuramoto-sivashinsky",
            category="physics",
            description="Kuramoto-Sivashinsky chaotic pattern equation",
            equations={"u": "-laplace(u) - nu * laplace(laplace(u)) - 0.5 * gradient_squared(u)"},
            parameters=[
                PDEParameter(
                    name="nu",
                    default=1.0,
                    description="Fourth-order diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Kuramoto-Sivashinsky spatiotemporal chaos",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Kuramoto-Sivashinsky equation PDE.

        Args:
            parameters: Dictionary containing 'nu' coefficient.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        nu = parameters.get("nu", 1.0)

        bc_spec = "periodic" if bc.get("x") == "periodic" else "no-flux"

        # py-pde uses gradient_squared for |grad(u)|^2
        return PDE(
            rhs={"u": f"-laplace(u) - {nu} * laplace(laplace(u)) - 0.5 * gradient_squared(u)"},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial state - typically small random perturbations."""
        if ic_type in ("kuramoto-sivashinsky-default", "default"):
            # Small random perturbations around zero
            amplitude = ic_params.get("amplitude", 0.1)
            np.random.seed(ic_params.get("seed"))
            data = amplitude * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        # Fall back to parent implementation
        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("kdv")
class KdVPDE(ScalarPDEPreset):
    """Korteweg-de Vries (KdV) equation.

    The KdV equation describes shallow water waves and solitons:

        du/dt = -c * d_dx(u) - alpha * u * d_dx(u) - beta * d_dx(d_dx(d_dx(u)))

    or more commonly written as:

        du/dt = -6 * u * d_dx(u) - d_dx(d_dx(d_dx(u)))

    where:
        - u is the wave amplitude
        - c is the wave speed
        - alpha is the nonlinear coefficient
        - beta is the dispersion coefficient

    Supports soliton solutions.

    Note: This is a 1D equation extended to 2D. For pure 1D behavior,
    use initial conditions that vary only in x.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kdv",
            category="physics",
            description="Korteweg-de Vries soliton equation",
            equations={"u": "-c * d_dx(u) - alpha * u * d_dx(u) - beta * d_dx(d_dx(d_dx(u)))"},
            parameters=[
                PDEParameter(
                    name="c",
                    default=0.0,
                    description="Linear wave speed",
                    min_value=-10.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=6.0,
                    description="Nonlinear coefficient",
                    min_value=0.0,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="beta",
                    default=1.0,
                    description="Dispersion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Korteweg & de Vries (1895)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 0.0)
        alpha = parameters.get("alpha", 6.0)
        beta = parameters.get("beta", 1.0)

        bc_spec = "periodic" if bc.get("x") == "periodic" else "no-flux"

        # Third derivative: d_dx(d_dx(d_dx(u)))
        # Linear term
        linear = f"-{c} * d_dx(u)" if c != 0 else ""
        # Nonlinear term
        nonlinear = f" - {alpha} * u * d_dx(u)"
        # Dispersion term
        dispersion = f" - {beta} * d_dx(d_dx(d_dx(u)))"

        rhs = (linear + nonlinear + dispersion).lstrip(" -")
        if rhs.startswith("- "):
            rhs = "-" + rhs[2:]

        return PDE(
            rhs={"u": rhs if rhs else "0"},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial state - soliton or wave packet."""
        if ic_type in ("kdv-default", "soliton"):
            # Soliton solution: u = A * sech^2(B*(x - x0))
            A = ic_params.get("amplitude", 1.0)
            width = ic_params.get("width", 0.1)
            x0 = ic_params.get("x0", 0.5)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            # sech^2 profile
            data = A / np.cosh((x - x0) / width) ** 2

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("ginzburg-landau")
class GinzburgLandauPDE(ScalarPDEPreset):
    """Complex Ginzburg-Landau equation.

    A universal model for pattern formation near instability:

        dA/dt = A + (1 + i*c1) * laplace(A) - (1 + i*c3) * |A|^2 * A

    where:
        - A is the complex amplitude
        - c1 is the linear dispersion coefficient
        - c3 is the nonlinear dispersion coefficient

    For c1 = c3 = 0, reduces to the real Ginzburg-Landau equation.
    Exhibits spiral waves, defect chaos, and other patterns.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="ginzburg-landau",
            category="physics",
            description="Complex Ginzburg-Landau pattern formation",
            equations={"A": "A + (1 + 1j*c1) * laplace(A) - (1 + 1j*c3) * |A|^2 * A"},
            parameters=[
                PDEParameter(
                    name="c1",
                    default=0.0,
                    description="Linear dispersion coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="c3",
                    default=0.0,
                    description="Nonlinear dispersion coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
            ],
            num_fields=1,
            field_names=["A"],
            reference="Ginzburg-Landau pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c1 = parameters.get("c1", 0.0)
        c3 = parameters.get("c3", 0.0)

        bc_spec = "periodic" if bc.get("x") == "periodic" else "no-flux"

        # Complex Ginzburg-Landau equation
        # Note: py-pde handles complex numbers with 1j notation
        return PDE(
            rhs={"A": f"A + (1 + 1j*{c1}) * laplace(A) - (1 + 1j*{c3}) * abs(A)**2 * A"},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial complex field with small perturbations."""
        if ic_type in ("ginzburg-landau-default", "default"):
            amplitude = ic_params.get("amplitude", 0.1)
            np.random.seed(ic_params.get("seed"))

            # Small complex perturbations
            real_part = amplitude * np.random.randn(*grid.shape)
            imag_part = amplitude * np.random.randn(*grid.shape)
            data = real_part + 1j * imag_part

            return ScalarField(grid, data, dtype=complex)

        field = create_initial_condition(grid, ic_type, ic_params)
        return ScalarField(grid, field.data.astype(complex), dtype=complex)
