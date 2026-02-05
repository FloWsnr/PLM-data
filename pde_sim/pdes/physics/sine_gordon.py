"""Sine-Gordon equation - topological kink solitons and breathers."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("sine-gordon")
class SineGordonPDE(MultiFieldPDEPreset):
    """Sine-Gordon equation - topological solitons and breathers.

    The Sine-Gordon equation is a fundamental nonlinear wave equation:

        d^2 phi / dt^2 = c^2 * laplace(phi) - sin(phi)

    Famous for supporting topological solitons (kinks and antikinks) that
    represent 2*pi phase transitions. In 2D, this becomes:

        d^2 phi / dt^2 = c^2 * (d^2 phi / dx^2 + d^2 phi / dy^2) - sin(phi)

    Converted to first-order system with optional damping:
        d(phi)/dt = psi + gamma * laplace(phi)
        d(psi)/dt = c^2 * laplace(phi) - sin(phi)

    where:
        - phi is the field variable (phase angle)
        - psi = d(phi)/dt is the velocity field
        - c is the wave speed
        - gamma is a stabilization/damping parameter

    Key phenomena:
        - Kink solitons: Topological transitions from 0 to 2*pi
        - Antikink solitons: Transitions from 2*pi to 0
        - Kink-antikink collisions: Can be elastic or create breathers
        - Breather solutions: Localized oscillating bound states
        - Ring solitons: Circular expanding/contracting waves in 2D

    Applications:
        - Josephson junctions in superconductors
        - Crystal dislocations
        - DNA dynamics
        - Particle physics (field theory models)
        - Magnetic flux quanta in thin films

    Reference: Perring & Skyrme (1962), Scott (1969), Ablowitz & Clarkson (1991)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="sine-gordon",
            category="physics",
            description="Sine-Gordon equation with topological kink solitons",
            equations={
                "phi": "psi + gamma * laplace(phi)",
                "psi": "c^2 * laplace(phi) - sin(phi)",
            },
            parameters=[
                PDEParameter("c", "Wave speed"),
                PDEParameter("gamma", "Damping/stabilization coefficient"),
            ],
            num_fields=2,
            field_names=["phi", "psi"],
            reference="Perring & Skyrme (1962), Scott (1969)",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Sine-Gordon equation PDE system.

        Args:
            parameters: Dictionary with c (wave speed) and gamma (damping).
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        c = parameters.get("c", 1.0)
        gamma = parameters.get("gamma", 0.01)
        c_sq = c * c

        bc_spec = self._convert_bc(bc)

        # Build phi equation with optional damping term
        if gamma > 0:
            phi_rhs = f"psi + {gamma} * laplace(phi)"
        else:
            phi_rhs = "psi"

        # Sine-Gordon: d^2(phi)/dt^2 = c^2 * laplace(phi) - sin(phi)
        # First-order system:
        #   d(phi)/dt = psi + gamma * laplace(phi)
        #   d(psi)/dt = c^2 * laplace(phi) - sin(phi)
        return PDE(
            rhs={
                "phi": phi_rhs,
                "psi": f"{c_sq} * laplace(phi) - sin(phi)",
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for Sine-Gordon equation.

        Supports several initial condition types for soliton dynamics:
        - kink: Single 1D kink soliton
        - antikink: Single 1D antikink soliton
        - kink-antikink: Kink-antikink pair for collision
        - breather: Oscillating bound state (breather)
        - ring: Circular ring soliton (2D specific)
        - random: Random initial perturbations
        """
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Get wave speed for soliton profiles
        c = kwargs.get("parameters", {}).get("c", 1.0)
        rng = np.random.default_rng(ic_params.get("seed"))

        if ic_type in ("default", "kink"):
            # Single kink soliton: phi = 4 * arctan(exp((x - x0) / w))
            # This transitions from 0 to 2*pi
            x0 = ic_params.get("x0")
            if x0 is None or x0 == "random":
                raise ValueError("sine-gordon kink requires x0 (or random)")
            width = ic_params.get("width", Lx * 0.05)
            velocity = ic_params.get("velocity", 0.0)

            # Lorentz factor for moving kink
            if abs(velocity) < c:
                gamma_lorentz = 1.0 / np.sqrt(1 - (velocity / c) ** 2)
            else:
                gamma_lorentz = 1.0

            # Kink profile
            phi_data = 4 * np.arctan(np.exp((X - x0) / (width / gamma_lorentz)))

            # Velocity field: psi = -velocity * d(phi)/dx for moving kink
            arg = (X - x0) / (width / gamma_lorentz)
            psi_data = -velocity * 2 * gamma_lorentz / (width * np.cosh(arg))

        elif ic_type == "antikink":
            # Antikink: transitions from 2*pi to 0
            x0 = ic_params.get("x0")
            if x0 is None or x0 == "random":
                raise ValueError("sine-gordon antikink requires x0 (or random)")
            width = ic_params.get("width", Lx * 0.05)
            velocity = ic_params.get("velocity", 0.0)

            if abs(velocity) < c:
                gamma_lorentz = 1.0 / np.sqrt(1 - (velocity / c) ** 2)
            else:
                gamma_lorentz = 1.0

            # Antikink profile (negative argument)
            phi_data = 4 * np.arctan(np.exp(-(X - x0) / (width / gamma_lorentz)))

            arg = -(X - x0) / (width / gamma_lorentz)
            psi_data = velocity * 2 * gamma_lorentz / (width * np.cosh(arg))

        elif ic_type == "kink-antikink":
            # Kink-antikink pair for collision dynamics
            x0_kink = ic_params.get("x0_kink")
            x0_antikink = ic_params.get("x0_antikink")
            if x0_kink is None or x0_kink == "random" or x0_antikink is None or x0_antikink == "random":
                raise ValueError("sine-gordon kink-antikink requires x0_kink and x0_antikink (or random)")
            width = ic_params.get("width", Lx * 0.05)
            v_kink = ic_params.get("v_kink", 0.3)
            v_antikink = ic_params.get("v_antikink", -0.3)

            # Kink moving right
            if abs(v_kink) < c:
                gamma_k = 1.0 / np.sqrt(1 - (v_kink / c) ** 2)
            else:
                gamma_k = 1.0
            w_k = width / gamma_k
            phi_kink = 4 * np.arctan(np.exp((X - x0_kink) / w_k))
            psi_kink = -v_kink * 2 * gamma_k / (width * np.cosh((X - x0_kink) / w_k))

            # Antikink moving left
            if abs(v_antikink) < c:
                gamma_ak = 1.0 / np.sqrt(1 - (v_antikink / c) ** 2)
            else:
                gamma_ak = 1.0
            w_ak = width / gamma_ak
            phi_antikink = 4 * np.arctan(np.exp(-(X - x0_antikink) / w_ak))
            psi_antikink = (
                v_antikink * 2 * gamma_ak / (width * np.cosh(-(X - x0_antikink) / w_ak))
            )

            # Combine (superposition works approximately for well-separated solitons)
            phi_data = phi_kink + phi_antikink - np.pi  # Subtract pi to center
            psi_data = psi_kink + psi_antikink

        elif ic_type == "breather":
            # Breather solution: localized oscillating bound state
            x0 = ic_params.get("x0")
            if x0 is None or x0 == "random":
                raise ValueError("sine-gordon breather requires x0 (or random)")
            width = ic_params.get("width", Lx * 0.1)
            omega = ic_params.get("omega", 0.5)  # Internal frequency

            # Initial phase can be set
            phase = ic_params.get("phase", np.pi / 2)

            # Breather profile at initial time
            arg = (X - x0) / width
            phi_data = 4 * np.arctan(np.sin(phase) / np.cosh(arg))
            # Time derivative: psi ~ omega * cos(phase) * sech
            psi_data = 4 * omega * np.cos(phase) / (np.cosh(arg) * (1 + np.sin(phase)**2 / np.cosh(arg)**2))

        elif ic_type == "ring":
            # Ring soliton: circular wave in 2D
            x_center = ic_params.get("x_center")
            y_center = ic_params.get("y_center")
            if x_center is None or x_center == "random" or y_center is None or y_center == "random":
                raise ValueError("sine-gordon ring requires x_center and y_center (or random)")
            R0 = ic_params.get("radius", min(Lx, Ly) * 0.25)
            width = ic_params.get("width", min(Lx, Ly) * 0.02)
            velocity = ic_params.get("velocity", 0.0)  # Radial velocity (expanding/contracting)

            r = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
            arg = (r - R0) / width

            # Ring soliton profile
            phi_data = 4 * np.arctan(np.exp(arg))

            # Radial velocity field
            psi_data = velocity * 2 / (width * np.cosh(arg))

        elif ic_type == "random":
            # Random small perturbations from phi = 0
            amplitude = ic_params.get("amplitude", 0.5)
            phi_data = amplitude * rng.standard_normal(grid.shape)
            psi_data = np.zeros(grid.shape)

        else:
            # Fall back to generic IC generator for phi, zero velocity for psi
            phi = create_initial_condition(grid, ic_type, ic_params)
            phi.label = "phi"

            psi_data = np.zeros(grid.shape)
            psi = ScalarField(grid, psi_data)
            psi.label = "psi"

            return FieldCollection([phi, psi])

        # Create fields
        phi_field = ScalarField(grid, phi_data)
        phi_field.label = "phi"
        psi_field = ScalarField(grid, psi_data)
        psi_field.label = "psi"

        return FieldCollection([phi_field, psi_field])

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        x_bounds = grid.axes_bounds[0]
        Lx = x_bounds[1] - x_bounds[0]
        y_bounds = grid.axes_bounds[1] if len(grid.axes_bounds) > 1 else None
        Ly = y_bounds[1] - y_bounds[0] if y_bounds is not None else None
        resolved = ic_params.copy()

        if ic_type in ("default", "kink", "antikink", "breather"):
            if "x0" not in resolved:
                raise ValueError(f"sine-gordon {ic_type} requires x0 (or random)")
            if resolved["x0"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                resolved["x0"] = rng.uniform(x_bounds[0] + 0.2 * Lx, x_bounds[0] + 0.8 * Lx)
            if resolved["x0"] is None:
                raise ValueError(f"sine-gordon {ic_type} requires x0 (or random)")
            return resolved

        if ic_type == "kink-antikink":
            if "x0_kink" not in resolved or "x0_antikink" not in resolved:
                raise ValueError("sine-gordon kink-antikink requires x0_kink and x0_antikink (or random)")
            if resolved["x0_kink"] == "random" or resolved["x0_antikink"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["x0_kink"] == "random":
                    resolved["x0_kink"] = rng.uniform(x_bounds[0] + 0.15 * Lx, x_bounds[0] + 0.4 * Lx)
                if resolved["x0_antikink"] == "random":
                    resolved["x0_antikink"] = rng.uniform(x_bounds[0] + 0.6 * Lx, x_bounds[0] + 0.85 * Lx)
            if resolved["x0_kink"] is None or resolved["x0_antikink"] is None:
                raise ValueError("sine-gordon kink-antikink requires x0_kink and x0_antikink (or random)")
            return resolved

        if ic_type == "ring":
            if "x_center" not in resolved or "y_center" not in resolved:
                raise ValueError("sine-gordon ring requires x_center and y_center (or random)")
            if resolved["x_center"] == "random" or resolved["y_center"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["x_center"] == "random":
                    resolved["x_center"] = rng.uniform(x_bounds[0] + 0.2 * Lx, x_bounds[0] + 0.8 * Lx)
                if resolved["y_center"] == "random":
                    if y_bounds is None or Ly is None:
                        raise ValueError("sine-gordon ring requires 2D grid for y_center")
                    resolved["y_center"] = rng.uniform(y_bounds[0] + 0.2 * Ly, y_bounds[0] + 0.8 * Ly)
            if resolved["x_center"] is None or resolved["y_center"] is None:
                raise ValueError("sine-gordon ring requires x_center and y_center (or random)")
            return resolved

        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        c = parameters.get("c", 1.0)
        gamma = parameters.get("gamma", 0.01)
        c_sq = c * c

        phi_eq = f"psi + {gamma} * laplace(phi)" if gamma > 0 else "psi"

        return {
            "phi": phi_eq,
            "psi": f"{c_sq} * laplace(phi) - sin(phi)",
        }
