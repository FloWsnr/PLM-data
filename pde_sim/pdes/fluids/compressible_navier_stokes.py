"""2D Compressible Navier-Stokes equations in primitive variable formulation."""

from typing import Any

import numpy as np
from pde import PDEBase, CartesianGrid, FieldCollection, ScalarField

from pde_sim.boundaries import BoundaryConditionFactory
from pde_sim.core.config import BoundaryConfig
from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class CompressibleNavierStokesPDEImpl(PDEBase):
    """Custom PDEBase for compressible Navier-Stokes with per-field boundary conditions.

    Primitive variable formulation with fields: rho (density), u (x-velocity),
    v (y-velocity), p (pressure).

    Equations (ideal gas, neglecting bulk viscosity):
        d(rho)/dt = -u * d_dx(rho) - v * d_dy(rho) - rho * (d_dx(u) + d_dy(v))
        d(u)/dt   = -u * d_dx(u) - v * d_dy(u) - (1/rho) * d_dx(p) + (mu/rho) * laplace(u)
        d(v)/dt   = -u * d_dx(v) - v * d_dy(v) - (1/rho) * d_dy(p) + (mu/rho) * laplace(v)
        d(p)/dt   = -u * d_dx(p) - v * d_dy(p) - gamma * p * (d_dx(u) + d_dy(v)) + kappa * laplace(p)
    """

    def __init__(self, gamma: float, mu: float, kappa: float,
                 bc_rho: Any, bc_u: Any, bc_v: Any, bc_p: Any):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
        self.kappa = kappa
        self.bc_rho = bc_rho
        self.bc_u = bc_u
        self.bc_v = bc_v
        self.bc_p = bc_p

    def evolution_rate(self, state, t=0):
        rho, u, v, p = state
        rhs = state.copy()

        # Gradients
        grad_rho = rho.gradient(self.bc_rho)
        grad_u = u.gradient(self.bc_u)
        grad_v = v.gradient(self.bc_v)
        grad_p = p.gradient(self.bc_p)

        # Divergence of velocity: d_dx(u) + d_dy(v)
        div_vel = grad_u[0] + grad_v[1]

        # Inverse density (element-wise)
        inv_rho = 1.0 / rho

        # Continuity: d(rho)/dt
        rhs[0] = -u * grad_rho[0] - v * grad_rho[1] - rho * div_vel

        # x-momentum: d(u)/dt
        rhs[1] = (-u * grad_u[0] - v * grad_u[1]
                  - inv_rho * grad_p[0]
                  + self.mu * inv_rho * u.laplace(self.bc_u))

        # y-momentum: d(v)/dt
        rhs[2] = (-u * grad_v[0] - v * grad_v[1]
                  - inv_rho * grad_p[1]
                  + self.mu * inv_rho * v.laplace(self.bc_v))

        # Pressure: d(p)/dt
        rhs[3] = (-u * grad_p[0] - v * grad_p[1]
                  - self.gamma * p * div_vel
                  + self.kappa * p.laplace(self.bc_p))

        return rhs


@register_pde("compressible-navier-stokes")
class CompressibleNavierStokesPDE(MultiFieldPDEPreset):
    """2D Compressible Navier-Stokes equations.

    Models compressible viscous fluid flow where density varies and pressure
    is governed by thermodynamics (ideal gas law). Unlike the incompressible
    formulation, this captures acoustic waves, shock formation, and
    compressibility effects.

    Fields: rho (density), u (x-velocity), v (y-velocity), p (pressure).

    Equations:
        d(rho)/dt = -u * d_dx(rho) - v * d_dy(rho) - rho * (d_dx(u) + d_dy(v))
        d(u)/dt   = -u * d_dx(u) - v * d_dy(u) - (1/rho) * d_dx(p) + (mu/rho) * laplace(u)
        d(v)/dt   = -u * d_dx(v) - v * d_dy(v) - (1/rho) * d_dy(p) + (mu/rho) * laplace(v)
        d(p)/dt   = -u * d_dx(p) - v * d_dy(p) - gamma * p * (d_dx(u) + d_dy(v)) + kappa * laplace(p)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="compressible-navier-stokes",
            category="fluids",
            description="2D compressible Navier-Stokes equations",
            equations={
                "rho": "-u * d_dx(rho) - v * d_dy(rho) - rho * (d_dx(u) + d_dy(v))",
                "u": "-u * d_dx(u) - v * d_dy(u) - (1/rho) * d_dx(p) + (mu/rho) * laplace(u)",
                "v": "-u * d_dx(v) - v * d_dy(v) - (1/rho) * d_dy(p) + (mu/rho) * laplace(v)",
                "p": "-u * d_dx(p) - v * d_dy(p) - gamma * p * (d_dx(u) + d_dy(v)) + kappa * laplace(p)",
            },
            parameters=[
                PDEParameter("gamma", "Ratio of specific heats"),
                PDEParameter("mu", "Dynamic viscosity"),
                PDEParameter("kappa", "Thermal/pressure diffusivity"),
            ],
            num_fields=4,
            field_names=["rho", "u", "v", "p"],
            reference="Compressible Navier-Stokes equations (ideal gas)",
            supported_dimensions=[2],
        )

    def _get_field_bc(
        self,
        bc: Any,
        field_name: str,
        parameters: dict[str, float] | None = None,
    ) -> Any:
        """Extract boundary condition for a specific field."""
        if isinstance(bc, BoundaryConfig):
            field_bc = bc.get_field_bc(field_name)
            return BoundaryConditionFactory.convert_field_bc(
                field_bc, parameters=parameters
            )
        elif isinstance(bc, dict):
            base_bc = {
                "x-": bc.get("x-", "periodic"),
                "x+": bc.get("x+", "periodic"),
                "y-": bc.get("y-", "periodic"),
                "y+": bc.get("y+", "periodic"),
            }
            if "fields" in bc and field_name in bc["fields"]:
                base_bc.update(bc["fields"][field_name])
            return BoundaryConditionFactory.convert_field_bc(
                base_bc, parameters=parameters
            )
        else:
            return bc

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: Any,
        grid: CartesianGrid,
    ) -> CompressibleNavierStokesPDEImpl:
        gamma = parameters.get("gamma", 1.4)
        mu = parameters.get("mu", 0.01)
        kappa = parameters.get("kappa", 0.01)

        bc_rho = self._get_field_bc(bc, "rho", parameters)
        bc_u = self._get_field_bc(bc, "u", parameters)
        bc_v = self._get_field_bc(bc, "v", parameters)
        bc_p = self._get_field_bc(bc, "p", parameters)

        return CompressibleNavierStokesPDEImpl(
            gamma=gamma,
            mu=mu,
            kappa=kappa,
            bc_rho=bc_rho,
            bc_u=bc_u,
            bc_v=bc_v,
            bc_p=bc_p,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for compressible Navier-Stokes.

        IC types:
            - acoustic-pulse / default: Gaussian pressure perturbation on uniform background
            - kelvin-helmholtz: Compressible shear layer with density stratification
            - density-blob: Gaussian density perturbation (isentropic)
            - shock-tube: 2D Riemann problem with left/right states
            - colliding-jets: Opposing horizontal streams that collide
        """
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Normalize coordinates to [0, 1]
        x_norm = (x - x_min) / L_x
        y_norm = (y - y_min) / L_y

        randomize = kwargs.get("randomize", False)

        if ic_type in ("compressible-navier-stokes-default", "default", "acoustic-pulse"):
            # Gaussian pressure perturbation on uniform background
            amplitude = ic_params.get("amplitude", 0.1)
            x0 = ic_params.get("x0")
            y0 = ic_params.get("y0")
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                x0 = rng.uniform(0.3, 0.7)
                y0 = rng.uniform(0.3, 0.7)
            if x0 is None or y0 is None:
                raise ValueError("acoustic-pulse requires x0 and y0")
            width = ic_params.get("width", 0.05)
            rho_0 = ic_params.get("rho_0", 1.0)
            p_0 = ic_params.get("p_0", 1.0)

            r_sq = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
            pulse = amplitude * np.exp(-r_sq / (2 * width**2))

            rho_data = rho_0 * np.ones_like(x)
            u_data = np.zeros_like(x)
            v_data = np.zeros_like(x)
            p_data = p_0 + pulse

        elif ic_type == "kelvin-helmholtz":
            # Compressible shear layer with density stratification
            shear_y = ic_params.get("shear_y")
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                shear_y = rng.uniform(0.35, 0.65)
            if shear_y is None:
                raise ValueError("kelvin-helmholtz requires shear_y")
            shear_width = ic_params.get("shear_width", 0.05)
            velocity_amplitude = ic_params.get("velocity_amplitude", 0.5)
            density_ratio = ic_params.get("density_ratio", 2.0)
            rho_0 = ic_params.get("rho_0", 1.0)
            p_0 = ic_params.get("p_0", 2.5)
            perturbation_amplitude = ic_params.get("perturbation_amplitude", 0.01)

            # Shear velocity profile (tanh)
            u_data = velocity_amplitude * np.tanh((y_norm - shear_y) / shear_width)
            # Perturbation to seed instability
            v_data = perturbation_amplitude * np.sin(2 * np.pi * x_norm) * np.exp(-((y_norm - shear_y) / (4 * shear_width)) ** 2)
            # Density jump across shear layer
            rho_data = rho_0 * (1.0 + (density_ratio - 1.0) * 0.5 * (1.0 + np.tanh((y_norm - shear_y) / shear_width)))
            # Uniform pressure (pressure equilibrium)
            p_data = p_0 * np.ones_like(x)

        elif ic_type == "density-blob":
            # Gaussian density perturbation (isentropic, pressure matched)
            amplitude = ic_params.get("amplitude", 0.5)
            x0 = ic_params.get("x0")
            y0 = ic_params.get("y0")
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                x0 = rng.uniform(0.3, 0.7)
                y0 = rng.uniform(0.3, 0.7)
            if x0 is None or y0 is None:
                raise ValueError("density-blob requires x0 and y0")
            width = ic_params.get("width", 0.08)
            rho_0 = ic_params.get("rho_0", 1.0)
            p_0 = ic_params.get("p_0", 1.0)

            r_sq = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
            blob = amplitude * np.exp(-r_sq / (2 * width**2))

            rho_data = rho_0 + blob
            u_data = np.zeros_like(x)
            v_data = np.zeros_like(x)
            # Isentropic: p/p_0 = (rho/rho_0)^gamma
            gamma = ic_params.get("gamma", 1.4)
            p_data = p_0 * (rho_data / rho_0) ** gamma

        elif ic_type == "shock-tube":
            # 2D shock tube (Riemann problem) with left/right states
            # Interface orientation: vertical (along x) or horizontal (along y)
            orientation = ic_params["orientation"]
            interface_pos = ic_params["interface_pos"]
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                interface_pos = rng.uniform(0.35, 0.65)
            rho_left = ic_params["rho_left"]
            rho_right = ic_params["rho_right"]
            p_left = ic_params["p_left"]
            p_right = ic_params["p_right"]
            u_left = ic_params.get("u_left", 0.0)
            u_right = ic_params.get("u_right", 0.0)
            v_left = ic_params.get("v_left", 0.0)
            v_right = ic_params.get("v_right", 0.0)
            # Smoothing width for the interface (avoids pure discontinuity)
            smooth = ic_params.get("smooth", 0.01)

            if orientation == "vertical":
                # Interface at x = interface_pos (normalized)
                transition = 0.5 * (1.0 + np.tanh((x_norm - interface_pos) / smooth))
            else:
                # Interface at y = interface_pos (normalized)
                transition = 0.5 * (1.0 + np.tanh((y_norm - interface_pos) / smooth))

            rho_data = rho_left + (rho_right - rho_left) * transition
            u_data = u_left + (u_right - u_left) * transition
            v_data = v_left + (v_right - v_left) * transition
            p_data = p_left + (p_right - p_left) * transition

        elif ic_type == "colliding-jets":
            # Two converging streams that collide at a horizontal stagnation line
            jet_y = ic_params["jet_y"]
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                jet_y = rng.uniform(0.35, 0.65)
            jet_width = ic_params["jet_width"]
            jet_velocity = ic_params["jet_velocity"]
            rho_0 = ic_params.get("rho_0", 1.0)
            p_0 = ic_params.get("p_0", 1.0)

            # Lower half flows upward, upper half flows downward → collision at jet_y
            transition = np.tanh((y_norm - jet_y) / jet_width)
            v_data = -jet_velocity * transition
            # Small x-perturbation to seed instability at the stagnation line
            u_data = 0.01 * np.sin(4 * np.pi * x_norm) * np.exp(-((y_norm - jet_y) / (4 * jet_width)) ** 2)
            rho_data = rho_0 * np.ones_like(x)
            p_data = p_0 * np.ones_like(x)

        else:
            # Fallback: use standard IC generator for rho, zeros for velocity, uniform pressure
            rho_field = create_initial_condition(grid, ic_type, ic_params, randomize=randomize)
            rho_data = rho_field.data
            u_data = np.zeros_like(rho_data)
            v_data = np.zeros_like(rho_data)
            p_data = np.ones_like(rho_data)

        rho = ScalarField(grid, rho_data)
        rho.label = "rho"
        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        p = ScalarField(grid, p_data)
        p.label = "p"

        return FieldCollection([rho, u, v, p])

