"""Schrodinger equation (quantum particle in a box)."""

import math
from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("schrodinger")
class SchrodingerPDE(MultiFieldPDEPreset):
    """Schrodinger equation (particle in a box) with optional potential.

    The Schrodinger equation describes quantum mechanical wave functions:

        i * dpsi/dt = -D * laplace(psi) + V(x,y) * psi

    In dimensionless form. Since the solver works with real numbers, we separate
    real (u) and imaginary (v) parts:

        du/dt = -D * laplace(v) + C*D * laplace(u) + V*v
        dv/dt = D * laplace(u) + C*D * laplace(v) - V*u

    where:
        - u = Re(psi), v = Im(psi)
        - D is the quantum mechanical parameter
        - C is an artificial diffusion parameter for numerical stabilization
        - V(x,y) is the potential energy function

    Supported potential types:
        - "none": Free particle (V=0)
        - "sinusoidal": V = V_strength * sin(pot_n*pi*x/L_x) * sin(pot_m*pi*y/L_y)
        - "harmonic": V = V_strength * ((x-cx)^2 + (y-cy)^2)

    Based on visualpde.com stabilized Schrodinger equation.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="schrodinger",
            category="basic",
            description="Schrodinger equation with optional potential",
            equations={
                "u": "-D * laplace(v) + C*D * laplace(u) + V*v",
                "v": "D * laplace(u) + C*D * laplace(v) - V*u",
            },
            parameters=[
                PDEParameter("D", "Quantum mechanical parameter"),
                PDEParameter("C", "Numerical stabilization parameter"),
                PDEParameter("n", "x wave number for initial eigenstate"),
                PDEParameter("m", "y wave number for initial eigenstate"),
                PDEParameter("potential_type", "Type of potential: none, sinusoidal, or harmonic"),
                PDEParameter("V_strength", "Potential amplitude (0 = no potential)"),
                PDEParameter("pot_n", "Potential mode number in x direction (sinusoidal)"),
                PDEParameter("pot_m", "Potential mode number in y direction (sinusoidal)"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/basic-pdes/stabilised-schrodinger",
            supported_dimensions=[1, 2, 3],
        )

    def _compute_potential(
        self,
        grid: CartesianGrid,
        potential_type: str,
        V_strength: float,
        pot_n: int,
        pot_m: int,
    ) -> ScalarField | None:
        """Compute the potential field V(x,y).

        Args:
            grid: The computational grid.
            potential_type: Type of potential ("none", "sinusoidal", "harmonic").
            V_strength: Amplitude of the potential.
            pot_n: x mode number (for sinusoidal potential).
            pot_m: y mode number (for sinusoidal potential).

        Returns:
            ScalarField with potential values, or None if potential_type is "none"
            or V_strength is 0.
        """
        if potential_type == "none" or V_strength == 0:
            return None

        # Get domain bounds
        L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

        # Get coordinates
        x_coords = grid.cell_coords[..., 0]
        y_coords = grid.cell_coords[..., 1]

        if potential_type == "sinusoidal":
            # V = V_strength * sin(pot_n*pi*x/L_x) * sin(pot_m*pi*y/L_y)
            V_data = V_strength * np.sin(pot_n * math.pi * x_coords / L_x) * np.sin(
                pot_m * math.pi * y_coords / L_y
            )
        elif potential_type == "harmonic":
            # V = V_strength * ((x-cx)^2 + (y-cy)^2)
            # Centered at domain center
            cx = (grid.axes_bounds[0][0] + grid.axes_bounds[0][1]) / 2
            cy = (grid.axes_bounds[1][0] + grid.axes_bounds[1][1]) / 2
            V_data = V_strength * ((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        else:
            raise ValueError(f"Unknown potential_type: {potential_type}")

        return ScalarField(grid, V_data)

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Schrodinger equation PDE.

        Args:
            parameters: Dictionary containing PDE coefficients and potential params.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters["D"]
        C = parameters["C"]

        # Potential parameters
        potential_type = parameters["potential_type"]
        V_strength = parameters["V_strength"]
        pot_n = int(parameters["pot_n"])
        pot_m = int(parameters["pot_m"])

        # Convert BC to py-pde format
        bc_spec = self._convert_bc(bc)

        # Compute potential field
        V_field = self._compute_potential(
            grid, potential_type, V_strength, pot_n, pot_m
        )

        # Build base equations
        # du/dt = -D * laplace(v) + C*D * laplace(u)
        # dv/dt = D * laplace(u) + C*D * laplace(v)
        if C > 0:
            base_u = f"-{D} * laplace(v) + {C * D} * laplace(u)"
            base_v = f"{D} * laplace(u) + {C * D} * laplace(v)"
        else:
            base_u = f"-{D} * laplace(v)"
            base_v = f"{D} * laplace(u)"

        # Add potential terms if present
        # du/dt = ... + V*v
        # dv/dt = ... - V*u
        if V_field is not None:
            u_rhs = f"{base_u} + V * v"
            v_rhs = f"{base_v} - V * u"
            consts = {"V": V_field}
        else:
            u_rhs = base_u
            v_rhs = base_v
            consts = {}

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs},
            bc=bc_spec,
            consts=consts if consts else None,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for Schrodinger equation.

        Default is an eigenstate sin(n*pi*x/L)*sin(m*pi*y/L).
        """
        if ic_type in ["eigenstate", "schrodinger-default"]:
            # Get domain bounds
            L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
            L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

            n = int(ic_params.get("n", 3))
            m = int(ic_params.get("m", 3))

            # Get coordinates
            x_coords = grid.cell_coords[..., 0]
            y_coords = grid.cell_coords[..., 1]

            # Eigenstate: sin(n*pi*x/L_x)*sin(m*pi*y/L_y)
            u_data = np.sin(n * np.pi * x_coords / L_x) * np.sin(
                m * np.pi * y_coords / L_y
            )
            v_data = np.zeros(grid.shape)  # Imaginary part starts at zero

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        elif ic_type == "superposition":
            # Superposition of two eigenstates for interesting beating dynamics.
            # A single eigenstate gives stationary |psi|^2; a superposition of
            # two modes with different energies produces time-varying density.
            L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
            L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

            n1 = int(ic_params["n1"])
            m1 = int(ic_params["m1"])
            n2 = int(ic_params["n2"])
            m2 = int(ic_params["m2"])
            weight = ic_params.get("weight", 0.5)

            x_coords = grid.cell_coords[..., 0]
            y_coords = grid.cell_coords[..., 1]

            mode1 = np.sin(n1 * np.pi * x_coords / L_x) * np.sin(
                m1 * np.pi * y_coords / L_y
            )
            mode2 = np.sin(n2 * np.pi * x_coords / L_x) * np.sin(
                m2 * np.pi * y_coords / L_y
            )

            u_data = (1 - weight) * mode1 + weight * mode2
            v_data = np.zeros(grid.shape)

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        elif ic_type == "wave-packet":
            # Gaussian wave packet
            x0 = ic_params.get("x0")
            y0 = ic_params.get("y0")
            if x0 is None or x0 == "random" or y0 is None or y0 == "random":
                raise ValueError("wave-packet requires x0 and y0 (or random)")
            sigma = ic_params.get("sigma", 0.1)
            kx = ic_params.get("kx", 3.0)
            ky = ic_params.get("ky", 0.0)

            # Get domain bounds for normalized coordinates
            L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
            L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

            # Get coordinates
            x_coords = grid.cell_coords[..., 0]
            y_coords = grid.cell_coords[..., 1]

            # Center of wave packet in absolute coordinates
            cx = grid.axes_bounds[0][0] + x0 * L_x
            cy = grid.axes_bounds[1][0] + y0 * L_y

            # Gaussian envelope with plane wave
            # psi = exp(-(r^2)/(2*sigma^2)) * exp(i*(kx*x + ky*y))
            r2 = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
            envelope = np.exp(-r2 / (2 * (sigma * L_x) ** 2))
            phase = kx * x_coords + ky * y_coords

            u_data = envelope * np.cos(phase)
            v_data = envelope * np.sin(phase)

            # Normalize
            norm = np.sqrt(np.sum(u_data**2 + v_data**2))
            if norm > 0:
                u_data = u_data / norm
                v_data = v_data / norm

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        elif ic_type == "localized":
            # Localized wave packet for use with potential
            # (sin(pi*x/L)*sin(pi*y/L))^10 - concentrated at center
            power = ic_params.get("power", 10)

            L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
            L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

            x_coords = grid.cell_coords[..., 0]
            y_coords = grid.cell_coords[..., 1]

            # Create localized state
            sin_x = np.sin(np.pi * x_coords / L_x)
            sin_y = np.sin(np.pi * y_coords / L_y)
            u_data = (sin_x * sin_y) ** power
            v_data = np.zeros(grid.shape)

            # Normalize
            norm = np.sqrt(np.sum(u_data**2 + v_data**2))
            if norm > 0:
                u_data = u_data / norm

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        # Fall back to parent implementation for standard IC types
        return super().create_initial_state(grid, ic_type, ic_params)

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type == "wave-packet":
            return {"x0", "y0"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type == "wave-packet":
            resolved = ic_params.copy()
            if "x0" not in resolved or "y0" not in resolved:
                raise ValueError("wave-packet requires x0 and y0 (or random)")
            if resolved["x0"] == "random" or resolved["y0"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["x0"] == "random":
                    resolved["x0"] = rng.uniform(0.2, 0.8)
                if resolved["y0"] == "random":
                    resolved["y0"] = rng.uniform(0.2, 0.8)
            if resolved["x0"] is None or resolved["y0"] is None:
                raise ValueError("wave-packet requires x0 and y0 (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters["D"]
        C = parameters["C"]
        V_strength = parameters["V_strength"]
        potential_type = parameters["potential_type"]

        if C > 0:
            base_u = f"-{D} * laplace(v) + {C*D} * laplace(u)"
            base_v = f"{D} * laplace(u) + {C*D} * laplace(v)"
        else:
            base_u = f"-{D} * laplace(v)"
            base_v = f"{D} * laplace(u)"

        if potential_type != "none" and V_strength > 0:
            return {
                "u": f"{base_u} + V*v",
                "v": f"{base_v} - V*u",
                "V(x,y)": f"{potential_type} potential with strength {V_strength}",
            }
        return {"u": base_u, "v": base_v}
