"""Schrodinger equation and plate equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("schrodinger")
class SchrodingerPDE(MultiFieldPDEPreset):
    """Schrodinger equation (particle in a box).

    The Schrodinger equation describes quantum mechanical wave functions:

        i * dpsi/dt = -D * laplace(psi)

    In dimensionless form. Since the solver works with real numbers, we separate
    real (u) and imaginary (v) parts:

        du/dt = -D * laplace(v) + C*D * laplace(u)
        dv/dt = D * laplace(u) + C*D * laplace(v)

    where:
        - u = Re(psi), v = Im(psi)
        - D is the quantum mechanical parameter
        - C is an artificial diffusion parameter for numerical stabilization

    Based on visualpde.com stabilized Schrodinger equation.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="schrodinger",
            category="basic",
            description="Schrodinger equation (quantum particle in a box)",
            equations={
                "u": "-D * laplace(v) + C*D * laplace(u)",
                "v": "D * laplace(u) + C*D * laplace(v)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Quantum mechanical parameter",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="C",
                    default=0.004,
                    description="Numerical stabilization parameter",
                    min_value=0.0,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="n",
                    default=3,
                    description="x wave number for initial eigenstate",
                    min_value=1,
                    max_value=10,
                ),
                PDEParameter(
                    name="m",
                    default=3,
                    description="y wave number for initial eigenstate",
                    min_value=1,
                    max_value=10,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/basic-pdes/schrodinger-equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Schrodinger equation PDE.

        Args:
            parameters: Dictionary containing 'D' and 'C' coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 1.0)
        C = parameters.get("C", 0.004)

        # Convert BC to py-pde format
        bc_spec = self._convert_bc(bc)

        # Build equations
        # du/dt = -D * laplace(v) + C*D * laplace(u)
        # dv/dt = D * laplace(u) + C*D * laplace(v)
        if C > 0:
            u_rhs = f"-{D} * laplace(v) + {C * D} * laplace(u)"
            v_rhs = f"{D} * laplace(u) + {C * D} * laplace(v)"
        else:
            u_rhs = f"-{D} * laplace(v)"
            v_rhs = f"{D} * laplace(u)"

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
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

        elif ic_type == "wave-packet":
            # Gaussian wave packet
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.5)
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

        # Fall back to parent implementation for standard IC types
        return super().create_initial_state(grid, ic_type, ic_params)


@register_pde("plate")
class PlatePDE(MultiFieldPDEPreset):
    """Plate vibration equation (biharmonic wave equation).

    The plate equation describes thin elastic plate vibrations using the
    biharmonic operator:

        d^2u/dt^2 = -D^2 * laplace(laplace(u)) - C * du/dt - Q

    Based on visualpde.com, reformulated using three fields for numerical stability:
        du/dt = v + D_c*D * laplace(u)
        dv/dt = -D * laplace(w) - C*v - Q
        w = D * laplace(u)  (auxiliary algebraic field)

    where:
        - u is plate displacement
        - v is velocity
        - w is an auxiliary field representing D * laplace(u)
        - D is the bending rigidity parameter
        - C is damping coefficient
        - Q is a constant load (gravity-like force)
        - D_c is a numerical stabilization parameter
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="plate",
            category="basic",
            description="Plate vibration equation (biharmonic wave)",
            equations={
                "u": "v + D_c*D * laplace(u)",
                "v": "-D * laplace(w) - C*v - Q",
                "w": "D * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=10.0,
                    description="Bending rigidity parameter",
                    min_value=0.1,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="Q",
                    default=0.003,
                    description="Constant load (gravity)",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="C",
                    default=0.1,
                    description="Damping coefficient",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="D_c",
                    default=0.1,
                    description="Numerical stabilization parameter",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="https://visualpde.com/basic-pdes/plate-equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the plate vibration PDE.

        Args:
            parameters: Dictionary containing 'D', 'Q', 'C', 'D_c' coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 10.0)
        Q = parameters.get("Q", 0.003)
        C = parameters.get("C", 0.1)
        D_c = parameters.get("D_c", 0.1)

        bc_spec = self._convert_bc(bc)

        # Build u equation: v + D_c*D * laplace(u)
        if D_c > 0:
            u_rhs = f"v + {D_c * D} * laplace(u)"
        else:
            u_rhs = "v"

        # Build v equation: -D * laplace(w) - C*v - Q
        v_terms = [f"-{D} * laplace(w)"]
        if C > 0:
            v_terms.append(f"- {C} * v")
        if Q != 0:
            v_terms.append(f"- {Q}")
        v_rhs = " ".join(v_terms)

        # w is computed from u: w = D * laplace(u)
        # We use cross-diffusion to implement this relationship
        w_rhs = f"{D} * laplace(u)"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": v_rhs,
                "w": w_rhs,
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for plate equation.

        Args:
            grid: The computational grid.
            ic_type: Type of initial condition.
            ic_params: Parameters for the initial condition.

        Returns:
            FieldCollection with u (displacement), v (velocity), and w (auxiliary) fields.
        """
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        # w starts at zero (will be computed from u during simulation)
        w_data = np.zeros(grid.shape)
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])
