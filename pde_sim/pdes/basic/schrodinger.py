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
        """Compute the potential field V(x,y,...).

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

        ndim = len(grid.shape)
        L = [grid.axes_bounds[i][1] - grid.axes_bounds[i][0] for i in range(ndim)]

        if potential_type == "sinusoidal":
            mode_numbers = [pot_n, pot_m, 1][:ndim]
            coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
            coords = np.meshgrid(*coords_1d, indexing="ij")
            V_data = V_strength * np.prod(
                [np.sin(mode_numbers[i] * math.pi * coords[i] / L[i]) for i in range(ndim)],
                axis=0,
            )
        elif potential_type == "harmonic":
            center = [(grid.axes_bounds[i][0] + grid.axes_bounds[i][1]) / 2 for i in range(ndim)]
            coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
            coords = np.meshgrid(*coords_1d, indexing="ij")
            r_sq = sum((coords[i] - center[i]) ** 2 for i in range(ndim))
            V_data = V_strength * r_sq
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
            ndim = len(grid.shape)
            L = [grid.axes_bounds[i][1] - grid.axes_bounds[i][0] for i in range(ndim)]
            mode_numbers_map = {"n": int(ic_params.get("n", 3)), "m": int(ic_params.get("m", 3)), "l": int(ic_params.get("l", 3))}
            mode_keys = ["n", "m", "l"][:ndim]
            mode_numbers = [mode_numbers_map[k] for k in mode_keys]

            coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
            coords = np.meshgrid(*coords_1d, indexing="ij")

            u_data = np.prod(
                [np.sin(mode_numbers[i] * np.pi * coords[i] / L[i]) for i in range(ndim)],
                axis=0,
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
            ndim = len(grid.shape)
            L = [grid.axes_bounds[i][1] - grid.axes_bounds[i][0] for i in range(ndim)]

            mode_keys_1 = ["n1", "m1", "l1"][:ndim]
            mode_keys_2 = ["n2", "m2", "l2"][:ndim]
            mode1_numbers = [int(ic_params[k]) for k in mode_keys_1]
            mode2_numbers = [int(ic_params[k]) for k in mode_keys_2]
            weight = ic_params.get("weight", 0.5)

            coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
            coords = np.meshgrid(*coords_1d, indexing="ij")

            mode1 = np.prod(
                [np.sin(mode1_numbers[i] * np.pi * coords[i] / L[i]) for i in range(ndim)],
                axis=0,
            )
            mode2 = np.prod(
                [np.sin(mode2_numbers[i] * np.pi * coords[i] / L[i]) for i in range(ndim)],
                axis=0,
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
            ndim = len(grid.shape)
            L = [grid.axes_bounds[i][1] - grid.axes_bounds[i][0] for i in range(ndim)]

            # Position params: x0, y0, z0
            pos_keys = ["x0", "y0", "z0"][:ndim]
            for key in pos_keys:
                val = ic_params.get(key)
                if val is None or val == "random":
                    raise ValueError(f"wave-packet requires {key} (or random)")

            sigma = ic_params.get("sigma", 0.1)
            k_keys = ["kx", "ky", "kz"][:ndim]
            k_defaults = [3.0, 0.0, 0.0]
            k_vals = [ic_params.get(k_keys[i], k_defaults[i]) for i in range(ndim)]

            coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
            coords = np.meshgrid(*coords_1d, indexing="ij")

            # Centers in absolute coordinates
            centers = [grid.axes_bounds[i][0] + ic_params[pos_keys[i]] * L[i] for i in range(ndim)]

            r2 = sum((coords[i] - centers[i]) ** 2 for i in range(ndim))
            envelope = np.exp(-r2 / (2 * (sigma * L[0]) ** 2))
            phase = sum(k_vals[i] * coords[i] for i in range(ndim))

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
            # Product of (sin(pi*x_i/L_i))^power - concentrated at center
            power = ic_params.get("power", 10)

            ndim = len(grid.shape)
            L = [grid.axes_bounds[i][1] - grid.axes_bounds[i][0] for i in range(ndim)]

            coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
            coords = np.meshgrid(*coords_1d, indexing="ij")

            # Create localized state: product of sin across all dims, raised to power
            u_data = np.prod(
                [np.sin(np.pi * coords[i] / L[i]) for i in range(ndim)],
                axis=0,
            ) ** power
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
            return {"x0", "y0", "z0"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type == "wave-packet":
            resolved = ic_params.copy()
            ndim = len(grid.shape)
            pos_keys = ["x0", "y0", "z0"][:ndim]
            for key in pos_keys:
                if key not in resolved:
                    raise ValueError(f"wave-packet requires {key} (or random)")
            rng_needed = any(resolved[key] == "random" for key in pos_keys)
            if rng_needed:
                rng = np.random.default_rng(resolved.get("seed"))
                for key in pos_keys:
                    if resolved[key] == "random":
                        resolved[key] = rng.uniform(0.2, 0.8)
            for key in pos_keys:
                if resolved[key] is None:
                    raise ValueError(f"wave-packet requires {key} (or random)")
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
