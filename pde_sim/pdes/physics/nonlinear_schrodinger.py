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
        if ic_type in ("nonlinear-schrodinger-default", "default", "soliton"):
            c = ic_params.get("c", 10.0)  # carrier wave velocity
            amplitude = ic_params.get("amplitude", 1.0)
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            # Get domain info
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            x0_frac = ic_params.get("x0_frac")
            if x0_frac is None or x0_frac == "random":
                raise ValueError("nonlinear-schrodinger soliton requires x0_frac (or random)")
            x0 = x_bounds[0] + Lx * x0_frac

            if len(grid.shape) > 1:
                y_bounds = grid.axes_bounds[1]
                y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
                X, Y = np.meshgrid(x, y, indexing="ij")

                # sech envelope
                envelope = amplitude / np.cosh(X - x0)

                # Carrier wave: cos(c*x), sin(c*x)
                u_data = np.cos(c * X) * envelope
                v_data = np.sin(c * X) * envelope
            else:
                envelope = amplitude / np.cosh(x - x0)
                u_data = np.cos(c * x) * envelope
                v_data = np.sin(c * x) * envelope

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
            x1_frac = ic_params.get("x1_frac")
            x2_frac = ic_params.get("x2_frac")
            if x1_frac is None or x1_frac == "random" or x2_frac is None or x2_frac == "random":
                raise ValueError("nonlinear-schrodinger two_soliton requires x1_frac and x2_frac (or random)")
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            # Get domain info
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])

            if len(grid.shape) > 1:
                y_bounds = grid.axes_bounds[1]
                y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
                X, Y = np.meshgrid(x, y, indexing="ij")

                # Two solitons at different positions
                x1 = x_bounds[0] + Lx * x1_frac
                x2 = x_bounds[0] + Lx * x2_frac

                # sech envelopes
                envelope1 = amplitude / np.cosh(X - x1)
                envelope2 = amplitude / np.cosh(X - x2)

                # Carrier waves with different velocities
                u_data = np.cos(c1 * X) * envelope1 + np.cos(c2 * X) * envelope2
                v_data = np.sin(c1 * X) * envelope1 + np.sin(c2 * X) * envelope2
            else:
                x1 = x_bounds[0] + Lx * x1_frac
                x2 = x_bounds[0] + Lx * x2_frac
                envelope1 = amplitude / np.cosh(x - x1)
                envelope2 = amplitude / np.cosh(x - x2)
                u_data = np.cos(c1 * x) * envelope1 + np.cos(c2 * x) * envelope2
                v_data = np.sin(c1 * x) * envelope1 + np.sin(c2 * x) * envelope2

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        # For other IC types
        return super().create_initial_state(grid, ic_type, ic_params)

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("nonlinear-schrodinger-default", "default", "soliton"):
            return {"x0_frac"}
        if ic_type == "two_soliton":
            return {"x1_frac", "x2_frac"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("nonlinear-schrodinger-default", "default", "soliton"):
            resolved = ic_params.copy()
            if "x0_frac" not in resolved:
                raise ValueError("nonlinear-schrodinger soliton requires x0_frac (or random)")
            if resolved["x0_frac"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                resolved["x0_frac"] = rng.uniform(0.1, 0.9)
            if resolved["x0_frac"] is None or resolved["x0_frac"] == "random":
                raise ValueError("nonlinear-schrodinger soliton requires x0_frac (or random)")
            return resolved

        if ic_type == "two_soliton":
            resolved = ic_params.copy()
            required = ["x1_frac", "x2_frac"]
            for key in required:
                if key not in resolved:
                    raise ValueError("nonlinear-schrodinger two_soliton requires x1_frac and x2_frac (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["x1_frac"] == "random" and resolved["x2_frac"] == "random":
                    vals = rng.uniform(0.1, 0.9, size=2)
                    vals.sort()
                    resolved["x1_frac"], resolved["x2_frac"] = vals.tolist()
                else:
                    if resolved["x1_frac"] == "random":
                        resolved["x1_frac"] = rng.uniform(0.1, 0.9)
                    if resolved["x2_frac"] == "random":
                        resolved["x2_frac"] = rng.uniform(0.1, 0.9)
            if any(resolved[key] is None or resolved[key] == "random" for key in required):
                raise ValueError("nonlinear-schrodinger two_soliton requires x1_frac and x2_frac (or random)")
            return resolved

        return super().resolve_ic_params(grid, ic_type, ic_params)

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
