"""Nonlinear Schrodinger equation for optical solitons."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("nonlinear-schrodinger")
class NonlinearSchrodingerPDE(MultiFieldPDEPreset):
    """Nonlinear Schrodinger (NLS) equation.

    Based on visualpde.com formulation:

        i * dpsi/dt = -laplace(psi) + kappa * psi * |psi|^2

    Or equivalently (multiplying by -i):
        dpsi/dt = i * laplace(psi) - i * kappa * psi * |psi|^2

    Separating into real and imaginary parts with psi = u + i*v:
        du/dt = -D * laplace(v) - kappa * v * (u^2 + v^2)
        dv/dt =  D * laplace(u) + kappa * u * (u^2 + v^2)

    A universal model for weakly nonlinear dispersive wave packets:
        - Fiber optics: soliton-based communication
        - Water waves: deep water wave packet dynamics
        - Bose-Einstein condensates: mean-field description
        - Plasma physics: Langmuir wave dynamics

    Key properties:
        - Complete integrability via Inverse Scattering Transform
        - Focusing (kappa > 0): bright solitons
        - Defocusing (kappa < 0): dark solitons

    Reference: Zakharov & Shabat (1972), Hasegawa & Tappert (1973)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="nonlinear-schrodinger",
            category="physics",
            description="Nonlinear Schrodinger for optical solitons",
            equations={
                "u": "-D * laplace(v) - kappa * v * (u**2 + v**2)",
                "v": "D * laplace(u) + kappa * u * (u**2 + v**2)",
            },
            parameters=[
                PDEParameter("kappa", "Nonlinearity parameter (>0: focusing, <0: defocusing)"),
                PDEParameter("D", "Dispersion coefficient"),
                PDEParameter("c", "Soliton velocity (for initial condition)"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Zakharov & Shabat (1972)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Nonlinear Schrodinger PDE.

        Args:
            parameters: Dictionary with kappa, D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        kappa = parameters.get("kappa", 1.0)
        D = parameters.get("D", 1.0)

        # NLS in real/imaginary form:
        # du/dt = -D * laplace(v) - kappa * v * (u^2 + v^2)
        # dv/dt =  D * laplace(u) + kappa * u * (u^2 + v^2)
        return PDE(
            rhs={
                "u": f"-{D} * laplace(v) - {kappa} * v * (u**2 + v**2)",
                "v": f"{D} * laplace(u) + {kappa} * u * (u**2 + v**2)",
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
        """Create initial state for NLS.

        Default: soliton profile with carrier wave.
        """
        randomize = kwargs.get("randomize", False)

        if ic_type in ("nonlinear-schrodinger-default", "default", "soliton"):
            c = ic_params.get("c", 10.0)  # carrier wave velocity
            amplitude = ic_params.get("amplitude", 1.0)

            # Get domain info
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]

            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                ic_params["x0_frac"] = rng.uniform(0.1, 0.9)

            x0_frac = ic_params.get("x0_frac")
            if x0_frac is None:
                raise ValueError("nonlinear-schrodinger soliton requires x0_frac")
            x0 = x_bounds[0] + Lx * x0_frac

            ndim = len(grid.shape)
            x_1d = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            shape = [1] * ndim
            shape[0] = grid.shape[0]
            X = x_1d.reshape(shape)

            # sech envelope (depends only on x)
            envelope = amplitude / np.cosh(X - x0)

            # Carrier wave: cos(c*x), sin(c*x)
            u_data = np.broadcast_to(np.cos(c * X) * envelope, grid.shape).copy()
            v_data = np.broadcast_to(np.sin(c * X) * envelope, grid.shape).copy()

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        elif ic_type == "two_soliton":
            # Two solitons moving toward each other for collision dynamics
            c1 = ic_params.get("c1", 8.0)  # velocity of first soliton
            c2 = ic_params.get("c2", -8.0)  # velocity of second soliton (opposite)
            amplitude = ic_params.get("amplitude", 1.0)

            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                vals = rng.uniform(0.1, 0.9, size=2)
                vals.sort()
                ic_params["x1_frac"], ic_params["x2_frac"] = vals.tolist()

            x1_frac = ic_params.get("x1_frac")
            x2_frac = ic_params.get("x2_frac")
            if x1_frac is None or x2_frac is None:
                raise ValueError("nonlinear-schrodinger two_soliton requires x1_frac and x2_frac")

            # Get domain info
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]
            x1 = x_bounds[0] + Lx * x1_frac
            x2 = x_bounds[0] + Lx * x2_frac

            ndim = len(grid.shape)
            x_1d = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            shape = [1] * ndim
            shape[0] = grid.shape[0]
            X = x_1d.reshape(shape)

            # sech envelopes (depend only on x)
            envelope1 = amplitude / np.cosh(X - x1)
            envelope2 = amplitude / np.cosh(X - x2)

            # Carrier waves with different velocities
            u_data = np.broadcast_to(np.cos(c1 * X) * envelope1 + np.cos(c2 * X) * envelope2, grid.shape).copy()
            v_data = np.broadcast_to(np.sin(c1 * X) * envelope1 + np.sin(c2 * X) * envelope2, grid.shape).copy()

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        # For other IC types
        return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        kappa = parameters.get("kappa", 1.0)
        D = parameters.get("D", 1.0)

        return {
            "u": f"-{D} * laplace(v) - {kappa} * v * (u**2 + v**2)",
            "v": f"{D} * laplace(u) + {kappa} * u * (u**2 + v**2)",
        }
